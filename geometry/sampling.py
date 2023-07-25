"""Module for bilinear sampling.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def bilinear_wrapper(imgs, coords):
    """Wrapper around bilinear sampling function, handles arbitrary input sizes.

    Args:
      imgs: [..., H_s, W_s, C] images to resample
      coords: [..., H_t, W_t, 2], source pixel locations from which to copy
    Returns:
      [..., H_t, W_t, C] images after bilinear sampling from input.
    """
    # The bilinear sampling code only handles 4D input, so we'll need to reshape.
    init_dims = imgs.get_shape().as_list()[:-3:]
    end_dims_img = imgs.get_shape().as_list()[-3::]
    end_dims_coords = coords.get_shape().as_list()[-3::]
    prod_init_dims = init_dims[0]
    for ix in range(1, len(init_dims)):
        prod_init_dims *= init_dims[ix]

    imgs = tf.reshape(imgs, [prod_init_dims] + end_dims_img)
    coords = tf.reshape(
        coords, [prod_init_dims] + end_dims_coords)
    imgs_sampled = tf.contrib.resampler.resampler(imgs, coords)
    imgs_sampled = tf.reshape(
        imgs_sampled, init_dims + imgs_sampled.get_shape().as_list()[-3::])
    return imgs_sampled
