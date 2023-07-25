"""TensorFlow utils for image transformations via translations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

import geometry.sampling


def divide_safe(num, den, name=None):
    eps = 1e-8
    den += eps * tf.cast(tf.equal(den, 0), 'float32')
    return tf.divide(num, den, name=name)


def inv_translation(translation, depths):
    """Computes inverse homography matrix between two cameras via a plane.

    Args:
        translation: [..., 3, 3], translations from source to target camera
        depths: [layers, batch]
    Returns:
        translation: [..., 3, 3] inverse translation matrices (translations mapping
          pixel coordinates from target to source).
    """
    with tf.name_scope('inv_translation'):
        n_layers, batch = depths.get_shape().as_list()

        # depths = depths()
        depths = tf.expand_dims(depths, axis=-1)
        depths = tf.expand_dims(depths, axis=-1)
        filler_1 = tf.constant([0.0, 0.0], shape=[1, 1, 1, 2])
        filler_1 = tf.tile(filler_1, [n_layers, batch, 1, 1])
        depths = tf.concat([filler_1, depths], axis=3)

        filler_2 = tf.constant([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], shape=[1, 1, 2, 3])
        filler_2 = tf.tile(filler_2, [n_layers, batch, 1, 1])
        depths = tf.concat([filler_2, depths], axis=2)

        inv_trans = tf.matmul(translation, depths, name='inv_trans')
        filler_3 = tf.constant([0.0, 0.0, 1.0], shape=[1, 1, 1, 3])
        filler_3 = tf.tile(filler_3, [n_layers, batch, 1, 1])
        inv_trans = tf.concat([inv_trans[:, :, :2, :], filler_3], axis=2)

        return inv_trans


def transform_points(points, homography):
    """Transforms input points according to homography.

    Args:
        points: [..., H, W, 3]; pixel (u,v,1) coordinates.
        homography: [..., 3, 3]; desired matrix transformation
    Returns:
        output_points: [..., H, W, 3]; transformed (u,v,w) coordinates.
    """
    with tf.name_scope('transform_points'):
        # Because the points have two additional dimensions as they vary across the
        # width and height of an image, we need to reshape to multiply by the
        # per-image homographies.
        points_orig_shape = points.get_shape().as_list()
        points_reshaped_shape = homography.get_shape().as_list()
        points_reshaped_shape[-2] = -1

        points_reshaped = tf.reshape(points, points_reshaped_shape)
        transformed_points = tf.matmul(points_reshaped, homography, transpose_b=True)
        transformed_points = tf.reshape(transformed_points, points_orig_shape)
        return transformed_points


def normalize_homogeneous(points):
    """Converts homogeneous coordinates to regular coordinates.

    Args:
        points: [..., n_dims_coords+1]; points in homogeneous coordinates.
    Returns:
        points_uv_norm: [..., n_dims_coords];
            points in standard coordinates after dividing by the last entry.
    """
    with tf.name_scope('normalize_homogeneous'):
        uv = points[..., :-1]
        w = tf.expand_dims(points[..., -1], -1)
        return divide_safe(uv, w)


def translate_plane_imgs(imgs, pixel_coords_trg, translation, depths):
    """Transforms input imgs via homographies for corresponding planes.

    Args:
      imgs: are [..., H_s, W_s, C]
      pixel_coords_trg: [..., H_t, W_t, 3]; pixel (u,v,1) coordinates.
      translation: [..., 3, 3], translations from source to target camera
      depths: [layers, batch]
    Returns:
      [..., H_t, W_t, C] images after bilinear sampling from input.
        Coordinates outside the image are sampled as 0.
    """
    with tf.name_scope('translate_plane_imgs'):
        trans_t2s_planes = inv_translation(translation, depths)
        pixel_coords_t2s = transform_points(pixel_coords_trg, trans_t2s_planes)
        pixel_coords_t2s = normalize_homogeneous(pixel_coords_t2s)
        imgs_s2t = geometry.sampling.bilinear_wrapper(imgs, pixel_coords_t2s)

        return imgs_s2t


def planar_translation(imgs, pixel_coords_trg, pose, depths):
    """Transforms imgs, masks and computes dmaps according to planar translation.

    Args:
      imgs: are [L, B, H, W, C], typically RGB images per layer
      pixel_coords_trg: tensors with shape [B, H_t, W_t, 3];
          pixel (u,v,1) coordinates of target image pixels. (typically meshgrid)
      pose: [B, 3, 3]
      depths: [layers, batch]
      Returns:
      imgs_transformed: [L, ..., C] images in trg frame
    Assumes the first dimension corresponds to layers.
    """
    with tf.name_scope('planar_translation'):
        n_layers = imgs.get_shape().as_list()[0]
        rot_rep_dims = [n_layers]
        rot_rep_dims += [1 for _ in range(len(pose.get_shape()))]

        cds_rep_dims = [n_layers]
        cds_rep_dims += [1 for _ in range(len(pixel_coords_trg.get_shape()))]

        translation = tf.tile(tf.expand_dims(pose, axis=0), rot_rep_dims)
        pixel_coords_trg = tf.tile(tf.expand_dims(
            pixel_coords_trg, axis=0), cds_rep_dims)

        imgs_trg = translate_plane_imgs(
            imgs, pixel_coords_trg, translation, depths)
        return imgs_trg
