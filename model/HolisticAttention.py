from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import scipy.stats as st


def gkern(kernlen=16, nsig=3):
    interval = (2 * nsig + 1.) / kernlen
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def min_max_norm(in_):
    max_ = tf.reduce_max(tf.reduce_max(in_, axis=2, keepdims=True), axis=1, keepdims=True)
    min_ = tf.reduce_min(tf.reduce_min(in_, axis=2, keepdims=True), axis=1, keepdims=True)
    in_ = in_ - min_
    return in_ / (max_ - min_ + 1e-8)


class HA(object):
    """Class definition for holistic attention module.
    """

    def __init__(self):
        gaussian_kernel = np.float32(gkern(31, 4))
        self.kernel_size = 31
        self.gaussian_kernel = gaussian_kernel[:, :, np.newaxis, np.newaxis]

    def net(self,
            attention,
            reuse=None,
            scope='net'):
        """holistic attention module
        Args:
        attention: output of the attention branch.
        reuse: whether or not the network and its variables should be reused. To be
          able to reuse 'scope' must be given.
        scope: Optional scope for the variables.
      Returns:
        soft_attention: soft attention map
      """
        with tf.variable_scope(scope, reuse=reuse):
            soft_attention = tf.layers.conv2d(attention, 1, self.kernel_size, padding='same', use_bias=False,
                                              kernel_initializer=tf.constant_initializer(value=self.gaussian_kernel),
                                              name='gaussian_kernel')
            soft_attention = min_max_norm(soft_attention)
            soft_attention = tf.maximum(soft_attention, attention)

            return soft_attention
