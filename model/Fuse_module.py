"""Functions for fusing features (some part from CVPR2019-CPD).
"""

from __future__ import division
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python import pywrap_tensorflow
import numpy as np

from HolisticAttention import HA


def RFB(inputs,
        out_channel,
        reuse=None,
        scope='net'):
    """receptive field block
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    out_channel: output channel of features.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional scope for the variables.
  Returns:
    x: the output of the RFB module
  """
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=tf.random_normal_initializer(stddev=0.01),
                            biases_initializer=tf.zeros_initializer(),
                            activation_fn=None):
            # branch0
            x0 = slim.conv2d(inputs, out_channel, [1, 1])

            # branch1
            x1 = slim.conv2d(inputs, out_channel, [1, 1])
            x1 = slim.conv2d(x1, out_channel, [1, 3])
            x1 = slim.conv2d(x1, out_channel, [3, 1])
            x1 = slim.conv2d(x1, out_channel, [3, 3], rate=3)

            # branch2
            x2 = slim.conv2d(inputs, out_channel, [1, 1])
            x2 = slim.conv2d(x2, out_channel, [1, 5])
            x2 = slim.conv2d(x2, out_channel, [5, 1])
            x2 = slim.conv2d(x2, out_channel, [3, 3], rate=5)

            # branch3
            x3 = slim.conv2d(inputs, out_channel, [1, 1])
            x3 = slim.conv2d(x3, out_channel, [1, 7])
            x3 = slim.conv2d(x3, out_channel, [7, 1])
            x3 = slim.conv2d(x3, out_channel, [3, 3], rate=7)

            x_cat = tf.concat([x0, x1, x2, x3], axis=3)
            x_cat = slim.conv2d(x_cat, out_channel, [3, 3])

            x = slim.conv2d(inputs, out_channel, [1, 1])
            x = tf.nn.relu(x_cat + x)

            return x


def aggregation(x1,
                x2,
                x3,
                out_channel,
                reuse=None,
                scope='net'):
    """aggregate the multi-level features
  Args:
    x1: 1/16, features of conv5.
    x2: 1/8, features of conv4.
    x3: 1/4, features of conv3.
    out_channel: output channel of features.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional scope for the variables.
  Returns:
    x: the output of the aggregation module
  """

    def upsample(im, scale_factor=2):
        im = tf.image.resize_images(im, [scale_factor * tf.shape(im)[1], scale_factor * tf.shape(im)[2]],
                                    align_corners=True)
        return im

    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=tf.random_normal_initializer(stddev=0.01),
                            biases_initializer=tf.zeros_initializer(),
                            activation_fn=None):
            x1_1 = x1
            x2_1 = slim.conv2d(upsample(x1), out_channel, [3, 3], scope='conv_upsample1') * x2
            x3_1 = slim.conv2d(upsample(upsample(x1)), out_channel, [3, 3], scope='conv_upsample2') \
                   * slim.conv2d(upsample(x2), out_channel, [3, 3], scope='conv_upsample3') * x3

            x2_2 = tf.concat([x2_1, slim.conv2d(upsample(x1_1), out_channel, [3, 3], scope='conv_upsample4')], axis=3)
            x2_2 = slim.conv2d(x2_2, 2 * out_channel, [3, 3], scope='conv_concat2')

            x3_2 = tf.concat([x3_1, slim.conv2d(upsample(x2_2), 2 * out_channel, [3, 3], scope='conv_upsample5')],
                             axis=3)
            x3_2 = slim.conv2d(x3_2, 3 * out_channel, [3, 3], scope='conv_concat3')

            x = slim.conv2d(x3_2, 3 * out_channel, [3, 3], scope='conv4')
            x = slim.conv2d(x, 1, [1, 1], scope='conv5')

            return x


def upsample(im, scale_factor=4):
    im = tf.image.resize_images(im, [scale_factor * tf.shape(im)[1], scale_factor * tf.shape(im)[2]],
                                align_corners=False)
    return im


def downsample(im, scale_factor=2):
    im = tf.image.resize_images(im, [tf.shape(im)[1] / scale_factor, tf.shape(im)[2] / scale_factor],
                                align_corners=True)
    return im


def min_max_norm(in_):
    max_ = tf.reduce_max(tf.reduce_max(in_, axis=2, keepdims=True), axis=1, keepdims=True)
    min_ = tf.reduce_min(tf.reduce_min(in_, axis=2, keepdims=True), axis=1, keepdims=True)
    in_ = in_ - min_
    return in_ / (max_ - min_ + 1e-8)


def weight_IOU(attention_rgb, attention_lf):
    Iand = tf.reduce_sum(tf.reduce_sum(attention_rgb * attention_lf, axis=2, keepdims=True), axis=1, keepdims=True)
    I_rgb = tf.reduce_sum(tf.reduce_sum(attention_rgb, axis=2, keepdims=True), axis=1, keepdims=True)
    I_lf = tf.reduce_sum(tf.reduce_sum(attention_lf, axis=2, keepdims=True), axis=1, keepdims=True)
    Ior = I_rgb + I_lf - Iand
    IoU = Iand / Ior

    return IoU


class CDIM(object):
    """Class definition for CDIM.
    """

    def __init__(self, channel=32):
        self.channel = channel
        self.HA = HA()
        pre_train = pywrap_tensorflow.NewCheckpointReader('./models/vgg16/vgg_16.ckpt')
        self._initialize_weights(pre_train)

    def fuse_module(self,
                    atts_rgb,
                    atts_lf,
                    feat_rgb,
                    feat_lf,
                    reuse=None,
                    scope='fuse_module'):
        """fuse module for two features
      Args:
        atts_rgb: attention map of rgb
        atts_lf: attention map of light field
        feat_rgb: feature of rgb (1/4)
        feat_lf: feature of light field (1/4)
        reuse: whether or not the network and its variables should be reused. To be
          able to reuse 'scope' must be given.
        scope: Optional scope for the variables.
      Returns:
        upsample(attention): the saliency map of attention branch.
        """
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.avg_pool2d]):
                # MCU
                attention = tf.concat([min_max_norm(tf.nn.sigmoid(atts_rgb)),
                                       min_max_norm(tf.nn.sigmoid(atts_lf))], axis=3)
                idx_lf = tf.to_float(tf.expand_dims(tf.argmax(attention, axis=3), -1))
                feat_max = (1 - idx_lf) * feat_rgb + idx_lf * feat_lf
                attention = tf.reduce_max(attention, axis=3, keepdims=True)
                atts_max = self.HA.net(attention, scope='HA', reuse=None)
                feat_mcu = atts_max * feat_max

                # IDU
                atts_tex = slim.max_pool2d(min_max_norm(tf.nn.sigmoid(atts_rgb)), [5, 5], stride=1,
                                           padding='SAME', scope='dilation_rgb')
                atts_pos = slim.max_pool2d(min_max_norm(tf.nn.sigmoid(atts_lf)), [5, 5], stride=1,
                                           padding='SAME', scope='dilation_lf')
                weight_iou = weight_IOU(atts_tex, atts_pos)
                atts_iou = weight_iou * atts_pos + (1 - weight_iou) * atts_tex
                feat_iou = atts_iou * feat_rgb

                x3 = feat_mcu + feat_iou
                with tf.variable_scope('conv4', reuse=reuse):
                    pool3 = slim.avg_pool2d(x3, [2, 2], scope='pool3')
                    conv4_1 = slim.conv2d(pool3, 512, [3, 3],
                                          weights_initializer=tf.constant_initializer(value=self.conv4_1_w),
                                          biases_initializer=tf.constant_initializer(value=self.conv4_1_b),
                                          scope='conv4_1')
                    conv4_2 = slim.conv2d(conv4_1, 512, [3, 3],
                                          weights_initializer=tf.constant_initializer(value=self.conv4_2_w),
                                          biases_initializer=tf.constant_initializer(value=self.conv4_2_b),
                                          scope='conv4_2')
                    x4 = slim.conv2d(conv4_2, 512, [3, 3],
                                     weights_initializer=tf.constant_initializer(value=self.conv4_3_w),
                                     biases_initializer=tf.constant_initializer(value=self.conv4_3_b),
                                     scope='conv4_3')

                with tf.variable_scope('conv5', reuse=reuse):
                    pool4 = slim.avg_pool2d(x4, [2, 2], scope='pool4')
                    conv5_1 = slim.conv2d(pool4, 512, [3, 3],
                                          weights_initializer=tf.constant_initializer(value=self.conv5_1_w),
                                          biases_initializer=tf.constant_initializer(value=self.conv5_1_b),
                                          scope='conv5_1')
                    conv5_2 = slim.conv2d(conv5_1, 512, [3, 3],
                                          weights_initializer=tf.constant_initializer(value=self.conv5_2_w),
                                          biases_initializer=tf.constant_initializer(value=self.conv5_2_b),
                                          scope='conv5_2')
                    x5 = slim.conv2d(conv5_2, 512, [3, 3],
                                     weights_initializer=tf.constant_initializer(value=self.conv5_3_w),
                                     biases_initializer=tf.constant_initializer(value=self.conv5_3_b),
                                     scope='conv5_3')

                return x3, x4, x5, upsample(attention)

    def decoder_module(self,
                       feat1,
                       feat2,
                       feat3,
                       reuse=None,
                       scope='fuse_module'):
        """decoder module for features
      Args:
        feat1: feature of fuse module (1/4)
        feat2: feature of fuse module (1/8)
        feat3: feature of fuse module (1/16)
        reuse: whether or not the network and its variables should be reused. To be
          able to reuse 'scope' must be given.
        scope: Optional scope for the variables.
      Returns:
        upsample(detection): the saliency map of detection branch.
        """
        with tf.variable_scope(scope, reuse=reuse):
            feat1 = RFB(feat1, self.channel, scope='rfb3', reuse=reuse)
            feat2 = RFB(feat2, self.channel, scope='rfb4', reuse=reuse)
            feat3 = RFB(feat3, self.channel, scope='rfb5', reuse=reuse)
            detection = aggregation(feat3, feat2, feat1, self.channel, scope='agg', reuse=reuse)

            return upsample(detection)

    def _initialize_weights(self, pretrain):
        self.conv4_1_w = pretrain.get_tensor('vgg_16/conv4/conv4_1/weights')
        self.conv4_2_w = pretrain.get_tensor('vgg_16/conv4/conv4_2/weights')
        self.conv4_3_w = pretrain.get_tensor('vgg_16/conv4/conv4_3/weights')
        self.conv5_1_w = pretrain.get_tensor('vgg_16/conv5/conv5_1/weights')
        self.conv5_2_w = pretrain.get_tensor('vgg_16/conv5/conv5_2/weights')
        self.conv5_3_w = pretrain.get_tensor('vgg_16/conv5/conv5_3/weights')

        self.conv4_1_b = pretrain.get_tensor('vgg_16/conv4/conv4_1/biases')
        self.conv4_2_b = pretrain.get_tensor('vgg_16/conv4/conv4_2/biases')
        self.conv4_3_b = pretrain.get_tensor('vgg_16/conv4/conv4_3/biases')
        self.conv5_1_b = pretrain.get_tensor('vgg_16/conv5/conv5_1/biases')
        self.conv5_2_b = pretrain.get_tensor('vgg_16/conv5/conv5_2/biases')
        self.conv5_3_b = pretrain.get_tensor('vgg_16/conv5/conv5_3/biases')
