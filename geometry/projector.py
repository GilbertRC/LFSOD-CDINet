"""A collection of projection utility functions.
"""
from __future__ import division
import tensorflow as tf
from geometry import translation

import numpy as np


def projective_forward_translation(src_images, pose, depths):
    """Use translation for forward warping.

    Args:
      src_images: [layers, batch, height, width, channels]
      pose: [batch, 3, 3]
      depths: [layers, batch]
    Returns:
      proj_src_images: [layers, batch, height, width, channels]
    """
    n_layers, n_batch, height, width, _ = src_images.get_shape().as_list()
    pixel_coords_trg = tf.transpose(
        meshgrid_abs(n_batch, height, width), [0, 2, 3, 1])
    proj_src_images = translation.planar_translation(
        src_images, pixel_coords_trg, pose, depths)
    return proj_src_images


def meshgrid_abs(batch, height, width, is_homogeneous=True):
    """Construct a 2D meshgrid in the absolute coordinates.

    Args:
      batch: batch size
      height: height of the grid
      width: width of the grid
      is_homogeneous: whether to return in homogeneous coordinates
    Returns:
      x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
    """
    xs = tf.linspace(0.0, tf.cast(width - 1, tf.float32), width)
    ys = tf.linspace(0.0, tf.cast(height - 1, tf.float32), height)
    xs, ys = tf.meshgrid(xs, ys)

    if is_homogeneous:
        ones = tf.ones_like(xs)
        coords = tf.stack([xs, ys, ones], axis=0)
    else:
        coords = tf.stack([xs, ys], axis=0)
    coords = tf.tile(tf.expand_dims(coords, 0), [batch, 1, 1, 1])
    return coords


def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
    """Transforms coordinates in the pixel frame to the camera frame.

    Args:
      depth: [batch, height, width]
      pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
      intrinsics: camera intrinsics [batch, 3, 3]
      is_homogeneous: return in homogeneous coordinates
    Returns:
      Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
    """

    batch, height, width = depth.get_shape().as_list()
    depth = tf.reshape(depth, [batch, 1, -1])

    pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
    cam_coords = tf.matmul(tf.matrix_inverse(intrinsics), pixel_coords) * depth
    if is_homogeneous:
        ones = tf.ones([batch, 1, height * width])
        cam_coords = tf.concat([cam_coords, ones], axis=1)
    cam_coords = tf.reshape(cam_coords, [batch, -1, height, width])
    return cam_coords


def cam2pixel(cam_coords, proj):
    """Transforms coordinates in a camera frame to the pixel frame.

    Args:
      cam_coords: [batch, 4, height, width]
      proj: [batch, 4, 4]
    Returns:
      Pixel coordinates projected from the camera frame [batch, height, width, 2]
    """
    batch, _, height, width = cam_coords.get_shape().as_list()
    cam_coords = tf.reshape(cam_coords, [batch, 4, -1])
    unnormalized_pixel_coords = tf.matmul(proj, cam_coords)

    xy_u = unnormalized_pixel_coords[:, 0:2, :]
    z_u = unnormalized_pixel_coords[:, 2:3, :]
    pixel_coords = xy_u / (z_u + 1e-10)
    pixel_coords = tf.reshape(pixel_coords, [batch, 2, height, width])

    return tf.transpose(pixel_coords, perm=[0, 2, 3, 1])


def psv_warp_lf(
        img, depth, pose):
    """Inverse warp a source image to the target image plane based on LF translation.

    Args:
      img: the source image [batch, height_s, width_s, 3]
      depth: depth map of the target image [batch, height_t, width_t]
      pose: target to source camera transformation matrix [batch, 3, 3]
    Returns:
      Source image inverse warped to the target image plane [batch, height_t,
      width_t, 3]
    """
    batch, height, width, _ = img.get_shape().as_list()
    # Construct pixel grid coordinates.
    pixel_coords = meshgrid_abs(batch, height, width)

    curr_depth = tf.constant([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, depth], shape=[1, 3, 3])
    curr_depth = tf.tile(curr_depth, [batch, 1, 1])
    proj_tgt_pixel_to_src_pixel = tf.matmul(pose, curr_depth)
    filler = tf.constant([0.0, 0.0, 1.0], shape=[1, 1, 3])
    filler = tf.tile(filler, [batch, 1, 1])
    proj_tgt_pixel_to_src_pixel = tf.concat([proj_tgt_pixel_to_src_pixel[:, :2, :], filler], axis=1)

    pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
    src_pixel_coords = tf.matmul(proj_tgt_pixel_to_src_pixel, pixel_coords)
    src_pixel_coords = src_pixel_coords[:, 0:2, :]
    src_pixel_coords = tf.reshape(src_pixel_coords, [batch, 2, height, width])
    src_pixel_coords = tf.transpose(src_pixel_coords, perm=[0, 2, 3, 1])

    output_img = tf.contrib.resampler.resampler(img, src_pixel_coords)

    return output_img


def over_composite(rgbas):
    """Combines a list of RGBA images using the over operation.

    Combines RGBA images from back to front with the over operation.
    The alpha image of the first image is ignored and assumed to be 1.0.

    Args:
      rgbas: A list of [batch, H, W, 4] RGBA images, combined from back to front.
    Returns:
      Composited RGB image.
    """
    for i in range(len(rgbas)):
        rgb = rgbas[i][:, :, :, 0:3]
        alpha = rgbas[i][:, :, :, 3:]
        if i == 0:
            output = rgb
            # output = rgb * alpha
        else:
            rgb_by_alpha = rgb * alpha
            output = rgb_by_alpha + output * (1.0 - alpha)
    return output


def plane_sweep(img, depth_planes, pose):
    """Construct a plane sweep volume.

    Args:
      img: source image [batch, height, width, #channels]
      depth_planes: a list of depth values for each plane
      pose: target to source camera transformation [batch, 3, 3]
    Returns:
      A plane sweep volume [batch, height, width, #planes*#channels]
    """
    batch, height, width, _ = img.get_shape().as_list()
    plane_sweep_volume = []

    for depth in depth_planes:
        warped_img = psv_warp_lf(img, depth, pose)
        plane_sweep_volume.append(warped_img)
    plane_sweep_volume = tf.concat(plane_sweep_volume, axis=3)
    return plane_sweep_volume
