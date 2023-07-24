"""Functions for learning rgb saliency (using VGG16).
"""
from __future__ import division
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python import pywrap_tensorflow
from Fuse_module import RFB
from Fuse_module import aggregation


class VGG(object):
    """Class definition for VGG learning module.
    """

    def __init__(self, channel=32):
        self.channel = channel
        pre_train = pywrap_tensorflow.NewCheckpointReader('./models/vgg16/vgg_16.ckpt')
        self._initialize_weights(pre_train)

    def vgg_net(self,
                inputs,
                reuse=None,
                scope='vgg_net'):
        """VGG16
      Args:
        inputs: a tensor of size [batch_size, height, width, channels].
        reuse: whether or not the network and its variables should be reused. To be
          able to reuse 'scope' must be given.
        scope: Optional scope for the variables.
      Returns:
        upsample(attention): the saliency map of attention branch.
        """
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.avg_pool2d]):
                # vgg_16
                with tf.variable_scope('conv1', reuse=reuse):
                    conv1_1 = slim.conv2d(inputs, 64, [3, 3],
                                          weights_initializer=tf.constant_initializer(value=self.conv1_1_w),
                                          biases_initializer=tf.constant_initializer(value=self.conv1_1_b),
                                          scope='conv1_1')
                    x1 = slim.conv2d(conv1_1, 64, [3, 3],
                                     weights_initializer=tf.constant_initializer(value=self.conv1_2_w),
                                     biases_initializer=tf.constant_initializer(value=self.conv1_2_b),
                                     scope='conv1_2')

                with tf.variable_scope('conv2', reuse=reuse):
                    pool1 = slim.avg_pool2d(x1, [2, 2], scope='pool1')
                    conv2_1 = slim.conv2d(pool1, 128, [3, 3],
                                          weights_initializer=tf.constant_initializer(value=self.conv2_1_w),
                                          biases_initializer=tf.constant_initializer(value=self.conv2_1_b),
                                          scope='conv2_1')
                    x2 = slim.conv2d(conv2_1, 128, [3, 3],
                                     weights_initializer=tf.constant_initializer(value=self.conv2_2_w),
                                     biases_initializer=tf.constant_initializer(value=self.conv2_2_b),
                                     scope='conv2_2')

                with tf.variable_scope('conv3', reuse=reuse):
                    pool2 = slim.avg_pool2d(x2, [2, 2], scope='pool2')
                    conv3_1 = slim.conv2d(pool2, 256, [3, 3],
                                          weights_initializer=tf.constant_initializer(value=self.conv3_1_w),
                                          biases_initializer=tf.constant_initializer(value=self.conv3_1_b),
                                          scope='conv3_1')
                    conv3_2 = slim.conv2d(conv3_1, 256, [3, 3],
                                          weights_initializer=tf.constant_initializer(value=self.conv3_2_w),
                                          biases_initializer=tf.constant_initializer(value=self.conv3_2_b),
                                          scope='conv3_2')
                    x3 = slim.conv2d(conv3_2, 256, [3, 3],
                                     weights_initializer=tf.constant_initializer(value=self.conv3_3_w),
                                     biases_initializer=tf.constant_initializer(value=self.conv3_3_b),
                                     scope='conv3_3')

                # attention branch
                x3_1 = x3
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

                x3 = RFB(x3, self.channel, scope='rfb3', reuse=reuse)
                x4 = RFB(x4, self.channel, scope='rfb4', reuse=reuse)
                x5 = RFB(x5, self.channel, scope='rfb5', reuse=reuse)
                attention = aggregation(x5, x4, x3, self.channel, scope='agg', reuse=reuse)

                return x3_1, attention

    def _initialize_weights(self, pretrain):
        self.conv1_1_w = pretrain.get_tensor('vgg_16/conv1/conv1_1/weights')
        self.conv1_2_w = pretrain.get_tensor('vgg_16/conv1/conv1_2/weights')
        self.conv2_1_w = pretrain.get_tensor('vgg_16/conv2/conv2_1/weights')
        self.conv2_2_w = pretrain.get_tensor('vgg_16/conv2/conv2_2/weights')
        self.conv3_1_w = pretrain.get_tensor('vgg_16/conv3/conv3_1/weights')
        self.conv3_2_w = pretrain.get_tensor('vgg_16/conv3/conv3_2/weights')
        self.conv3_3_w = pretrain.get_tensor('vgg_16/conv3/conv3_3/weights')
        self.conv4_1_w = pretrain.get_tensor('vgg_16/conv4/conv4_1/weights')
        self.conv4_2_w = pretrain.get_tensor('vgg_16/conv4/conv4_2/weights')
        self.conv4_3_w = pretrain.get_tensor('vgg_16/conv4/conv4_3/weights')
        self.conv5_1_w = pretrain.get_tensor('vgg_16/conv5/conv5_1/weights')
        self.conv5_2_w = pretrain.get_tensor('vgg_16/conv5/conv5_2/weights')
        self.conv5_3_w = pretrain.get_tensor('vgg_16/conv5/conv5_3/weights')

        self.conv1_1_b = pretrain.get_tensor('vgg_16/conv1/conv1_1/biases')
        self.conv1_2_b = pretrain.get_tensor('vgg_16/conv1/conv1_2/biases')
        self.conv2_1_b = pretrain.get_tensor('vgg_16/conv2/conv2_1/biases')
        self.conv2_2_b = pretrain.get_tensor('vgg_16/conv2/conv2_2/biases')
        self.conv3_1_b = pretrain.get_tensor('vgg_16/conv3/conv3_1/biases')
        self.conv3_2_b = pretrain.get_tensor('vgg_16/conv3/conv3_2/biases')
        self.conv3_3_b = pretrain.get_tensor('vgg_16/conv3/conv3_3/biases')
        self.conv4_1_b = pretrain.get_tensor('vgg_16/conv4/conv4_1/biases')
        self.conv4_2_b = pretrain.get_tensor('vgg_16/conv4/conv4_2/biases')
        self.conv4_3_b = pretrain.get_tensor('vgg_16/conv4/conv4_3/biases')
        self.conv5_1_b = pretrain.get_tensor('vgg_16/conv5/conv5_1/biases')
        self.conv5_2_b = pretrain.get_tensor('vgg_16/conv5/conv5_2/biases')
        self.conv5_3_b = pretrain.get_tensor('vgg_16/conv5/conv5_3/biases')
