"""Functions for learning light field saliency (using MPI networks).
"""
from __future__ import division
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python import pywrap_tensorflow
import geometry.projector as pj
from model.Fuse_module import RFB
from model.Fuse_module import aggregation
from model.Fuse_module import downsample


class MPI(object):
    """Class definition for MPI learning module.
    """

    def __init__(self, channel=32, is_trainable=True):
        self.channel = channel
        self.is_trainable = is_trainable
        pre_train = pywrap_tensorflow.NewCheckpointReader('./models/mpi/siggraph_model'
                                                          '/model.latest')
        self._initialize_weights(pre_train)

    def mpi_net(self,
                inputs,
                ngf=64,
                scope='mpi_net',
                reuse=False,
                num_outputs=32):
        """Network definition for extracting features from multiplane image (MPI) inference.

      Args:
        inputs: stack of input images [batch, height, width, input_channels]
        ngf: number of features for the first conv layer
        scope: variable scope
        reuse: whether to reuse weights (for weight sharing)
        num_outputs: number of output channels for the predicted mpi
      Returns:
        pred: network output at the same spatial resolution as the inputs.
        """
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope(
                    [slim.conv2d, slim.conv2d_transpose], trainable=self.is_trainable,
                    normalizer_fn=slim.layer_norm_initialized):

                # first layer start from 5 views, thus trained from default parameters
                cnv1_1 = slim.conv2d(inputs, ngf, [3, 3], scope='conv1_1', stride=1,
                                     normalizer_params={'trainable': self.is_trainable})
                cnv1_2 = slim.conv2d(cnv1_1, ngf * 2, [3, 3], scope='conv1_2', stride=2,
                                     normalizer_params={'trainable': self.is_trainable})

                cnv2_1 = slim.conv2d(cnv1_2, ngf * 2, [3, 3], scope='conv2_1', stride=1,
                                     weights_initializer=tf.constant_initializer(value=self.conv2_1_w),
                                     normalizer_params={
                                         'beta_initializer': tf.constant_initializer(value=self.conv2_1_beta),
                                         'gamma_initializer': tf.constant_initializer(value=self.conv2_1_gamma),
                                         'trainable': self.is_trainable})
                cnv2_2 = slim.conv2d(cnv2_1, ngf * 4, [3, 3], scope='conv2_2', stride=2,
                                     weights_initializer=tf.constant_initializer(value=self.conv2_2_w),
                                     normalizer_params={
                                         'beta_initializer': tf.constant_initializer(value=self.conv2_2_beta),
                                         'gamma_initializer': tf.constant_initializer(value=self.conv2_2_gamma),
                                         'trainable': self.is_trainable})

                cnv3_1 = slim.conv2d(cnv2_2, ngf * 4, [3, 3], scope='conv3_1', stride=1,
                                     weights_initializer=tf.constant_initializer(value=self.conv3_1_w),
                                     normalizer_params={
                                         'beta_initializer': tf.constant_initializer(value=self.conv3_1_beta),
                                         'gamma_initializer': tf.constant_initializer(value=self.conv3_1_gamma),
                                         'trainable': self.is_trainable})
                cnv3_2 = slim.conv2d(cnv3_1, ngf * 4, [3, 3], scope='conv3_2', stride=1,
                                     weights_initializer=tf.constant_initializer(value=self.conv3_2_w),
                                     normalizer_params={
                                         'beta_initializer': tf.constant_initializer(value=self.conv3_2_beta),
                                         'gamma_initializer': tf.constant_initializer(value=self.conv3_2_gamma),
                                         'trainable': self.is_trainable})
                cnv3_3 = slim.conv2d(cnv3_2, ngf * 8, [3, 3], scope='conv3_3', stride=2,
                                     weights_initializer=tf.constant_initializer(value=self.conv3_3_w),
                                     normalizer_params={
                                         'beta_initializer': tf.constant_initializer(value=self.conv3_3_beta),
                                         'gamma_initializer': tf.constant_initializer(value=self.conv3_3_gamma),
                                         'trainable': self.is_trainable})

                cnv4_1 = slim.conv2d(cnv3_3, ngf * 8, [3, 3], scope='conv4_1', stride=1, rate=2,
                                     weights_initializer=tf.constant_initializer(value=self.conv4_1_w),
                                     normalizer_params={
                                         'beta_initializer': tf.constant_initializer(value=self.conv4_1_beta),
                                         'gamma_initializer': tf.constant_initializer(value=self.conv4_1_gamma),
                                         'trainable': self.is_trainable})
                cnv4_2 = slim.conv2d(cnv4_1, ngf * 8, [3, 3], scope='conv4_2', stride=1, rate=2,
                                     weights_initializer=tf.constant_initializer(value=self.conv4_2_w),
                                     normalizer_params={
                                         'beta_initializer': tf.constant_initializer(value=self.conv4_2_beta),
                                         'gamma_initializer': tf.constant_initializer(value=self.conv4_2_gamma),
                                         'trainable': self.is_trainable})
                cnv4_3 = slim.conv2d(cnv4_2, ngf * 8, [3, 3], scope='conv4_3', stride=1, rate=2,
                                     weights_initializer=tf.constant_initializer(value=self.conv4_3_w),
                                     normalizer_params={
                                         'beta_initializer': tf.constant_initializer(value=self.conv4_3_beta),
                                         'gamma_initializer': tf.constant_initializer(value=self.conv4_3_gamma),
                                         'trainable': self.is_trainable})

                # Adding skips
                skip = tf.concat([cnv4_3, cnv3_3], axis=3)
                feat3 = skip
                cnv5_1 = slim.conv2d_transpose(skip, ngf * 4, [4, 4], scope='conv5_1', stride=2,
                                               weights_initializer=tf.constant_initializer(value=self.conv5_1_w),
                                               normalizer_params={
                                                   'beta_initializer': tf.constant_initializer(value=self.conv5_1_beta),
                                                   'gamma_initializer': tf.constant_initializer(
                                                       value=self.conv5_1_gamma),
                                                   'trainable': self.is_trainable})
                cnv5_2 = slim.conv2d(cnv5_1, ngf * 4, [3, 3], scope='conv5_2', stride=1,
                                     weights_initializer=tf.constant_initializer(value=self.conv5_2_w),
                                     normalizer_params={
                                         'beta_initializer': tf.constant_initializer(value=self.conv5_2_beta),
                                         'gamma_initializer': tf.constant_initializer(value=self.conv5_2_gamma),
                                         'trainable': self.is_trainable})
                cnv5_3 = slim.conv2d(cnv5_2, ngf * 4, [3, 3], scope='conv5_3', stride=1,
                                     weights_initializer=tf.constant_initializer(value=self.conv5_3_w),
                                     normalizer_params={
                                         'beta_initializer': tf.constant_initializer(value=self.conv5_3_beta),
                                         'gamma_initializer': tf.constant_initializer(value=self.conv5_3_gamma),
                                         'trainable': self.is_trainable})

                skip = tf.concat([cnv5_3, cnv2_2], axis=3)
                feat2 = skip
                cnv6_1 = slim.conv2d_transpose(skip, ngf * 2, [4, 4], scope='conv6_1', stride=2,
                                               weights_initializer=tf.constant_initializer(value=self.conv6_1_w),
                                               normalizer_params={
                                                   'beta_initializer': tf.constant_initializer(value=self.conv6_1_beta),
                                                   'gamma_initializer': tf.constant_initializer(
                                                       value=self.conv6_1_gamma),
                                                   'trainable': self.is_trainable})
                cnv6_2 = slim.conv2d(cnv6_1, ngf * 2, [3, 3], scope='conv6_2', stride=1,
                                     weights_initializer=tf.constant_initializer(value=self.conv6_2_w),
                                     normalizer_params={
                                         'beta_initializer': tf.constant_initializer(value=self.conv6_2_beta),
                                         'gamma_initializer': tf.constant_initializer(value=self.conv6_2_gamma),
                                         'trainable': self.is_trainable})

                skip = tf.concat([cnv6_2, cnv1_2], axis=3)
                feat1 = skip

                # view synthesis
                cnv7_1 = slim.conv2d_transpose(skip, ngf, [4, 4], scope='conv7_1', stride=2,
                                               weights_initializer=tf.constant_initializer(value=self.conv7_1_w),
                                               normalizer_params={
                                                   'beta_initializer': tf.constant_initializer(
                                                       value=self.conv7_1_beta),
                                                   'gamma_initializer': tf.constant_initializer(
                                                       value=self.conv7_1_gamma),
                                                   'trainable': self.is_trainable})
                cnv7_2 = slim.conv2d(cnv7_1, ngf, [3, 3], scope='conv7_2', stride=1,
                                     weights_initializer=tf.constant_initializer(value=self.conv7_2_w),
                                     normalizer_params={
                                         'beta_initializer': tf.constant_initializer(value=self.conv7_2_beta),
                                         'gamma_initializer': tf.constant_initializer(value=self.conv7_2_gamma),
                                         'trainable': self.is_trainable})

                pred = slim.conv2d(cnv7_2, num_outputs, [1, 1], stride=1, activation_fn=tf.nn.tanh,
                                   normalizer_fn=None, scope='mpi_pred')

            # detection feature
            feat_lf1 = downsample(feat1)
            feat1 = RFB(downsample(feat1), self.channel, scope='rfb3', reuse=reuse)
            feat2 = RFB(downsample(feat2), self.channel, scope='rfb4', reuse=reuse)
            feat3 = RFB(downsample(feat3), self.channel, scope='rfb5', reuse=reuse)
            attention_lf = aggregation(feat3, feat2, feat1, self.channel, scope='agg', reuse=reuse)

            return feat_lf1, attention_lf, pred

    def _initialize_weights(self, pretrain):
        # all_variables = pretrain.get_variable_to_shape_map()
        # self.conv1_1_w = pretrain.get_tensor('net/conv1_1/weights')
        # self.conv1_2_w = pretrain.get_tensor('net/conv1_2/weights')
        self.conv2_1_w = pretrain.get_tensor('net/conv2_1/weights')
        self.conv2_2_w = pretrain.get_tensor('net/conv2_2/weights')
        self.conv3_1_w = pretrain.get_tensor('net/conv3_1/weights')
        self.conv3_2_w = pretrain.get_tensor('net/conv3_2/weights')
        self.conv3_3_w = pretrain.get_tensor('net/conv3_3/weights')
        self.conv4_1_w = pretrain.get_tensor('net/conv4_1/weights')
        self.conv4_2_w = pretrain.get_tensor('net/conv4_2/weights')
        self.conv4_3_w = pretrain.get_tensor('net/conv4_3/weights')
        self.conv5_1_w = pretrain.get_tensor('net/conv6_1/weights')
        self.conv5_2_w = pretrain.get_tensor('net/conv6_2/weights')
        self.conv5_3_w = pretrain.get_tensor('net/conv6_3/weights')
        self.conv6_1_w = pretrain.get_tensor('net/conv7_1/weights')
        self.conv6_2_w = pretrain.get_tensor('net/conv7_2/weights')
        self.conv7_1_w = pretrain.get_tensor('net/conv8_1/weights')
        self.conv7_2_w = pretrain.get_tensor('net/conv8_2/weights')

        # self.conv1_1_gamma = pretrain.get_tensor('net/conv1_1/LayerNorm/gamma')
        # self.conv1_2_gamma = pretrain.get_tensor('net/conv1_2/LayerNorm/gamma')
        self.conv2_1_gamma = pretrain.get_tensor('net/conv2_1/LayerNorm/gamma')
        self.conv2_2_gamma = pretrain.get_tensor('net/conv2_2/LayerNorm/gamma')
        self.conv3_1_gamma = pretrain.get_tensor('net/conv3_1/LayerNorm/gamma')
        self.conv3_2_gamma = pretrain.get_tensor('net/conv3_2/LayerNorm/gamma')
        self.conv3_3_gamma = pretrain.get_tensor('net/conv3_3/LayerNorm/gamma')
        self.conv4_1_gamma = pretrain.get_tensor('net/conv4_1/LayerNorm/gamma')
        self.conv4_2_gamma = pretrain.get_tensor('net/conv4_2/LayerNorm/gamma')
        self.conv4_3_gamma = pretrain.get_tensor('net/conv4_3/LayerNorm/gamma')
        self.conv5_1_gamma = pretrain.get_tensor('net/conv6_1/LayerNorm/gamma')
        self.conv5_2_gamma = pretrain.get_tensor('net/conv6_2/LayerNorm/gamma')
        self.conv5_3_gamma = pretrain.get_tensor('net/conv6_3/LayerNorm/gamma')
        self.conv6_1_gamma = pretrain.get_tensor('net/conv7_1/LayerNorm/gamma')
        self.conv6_2_gamma = pretrain.get_tensor('net/conv7_2/LayerNorm/gamma')
        self.conv7_1_gamma = pretrain.get_tensor('net/conv8_1/LayerNorm/gamma')
        self.conv7_2_gamma = pretrain.get_tensor('net/conv8_2/LayerNorm/gamma')

        # self.conv1_1_beta = pretrain.get_tensor('net/conv1_1/LayerNorm/beta')
        # self.conv1_2_beta = pretrain.get_tensor('net/conv1_2/LayerNorm/beta')
        self.conv2_1_beta = pretrain.get_tensor('net/conv2_1/LayerNorm/beta')
        self.conv2_2_beta = pretrain.get_tensor('net/conv2_2/LayerNorm/beta')
        self.conv3_1_beta = pretrain.get_tensor('net/conv3_1/LayerNorm/beta')
        self.conv3_2_beta = pretrain.get_tensor('net/conv3_2/LayerNorm/beta')
        self.conv3_3_beta = pretrain.get_tensor('net/conv3_3/LayerNorm/beta')
        self.conv4_1_beta = pretrain.get_tensor('net/conv4_1/LayerNorm/beta')
        self.conv4_2_beta = pretrain.get_tensor('net/conv4_2/LayerNorm/beta')
        self.conv4_3_beta = pretrain.get_tensor('net/conv4_3/LayerNorm/beta')
        self.conv5_1_beta = pretrain.get_tensor('net/conv6_1/LayerNorm/beta')
        self.conv5_2_beta = pretrain.get_tensor('net/conv6_2/LayerNorm/beta')
        self.conv5_3_beta = pretrain.get_tensor('net/conv6_3/LayerNorm/beta')
        self.conv6_1_beta = pretrain.get_tensor('net/conv7_1/LayerNorm/beta')
        self.conv6_2_beta = pretrain.get_tensor('net/conv7_2/LayerNorm/beta')
        self.conv7_1_beta = pretrain.get_tensor('net/conv8_1/LayerNorm/beta')
        self.conv7_2_beta = pretrain.get_tensor('net/conv8_2/LayerNorm/beta')

    def infer_mpi(self,
                  mpi_net_input,
                  num_mpi_planes,
                  is_render=True,
                  reuse=False):
        """Construct the MPI inference graph.

        Args:
          mpi_net_input: stack of input images [batch, height, width, input_channels]
          num_mpi_planes: number of MPI planes to predict
          is_render: whether to render a new view
          reuse: whether to reuse weights (for weight sharing)
        Returns:
          outputs: a collection of output tensors.
        """
        batch_size, img_height, img_width, _ = mpi_net_input.get_shape().as_list()
        # No color image (or blending weights) is predicted by the network.
        # The reference source image is used as the color image at each MPI plane.
        if is_render:
            feat_lf, atts_lf, alphas = self.mpi_net(mpi_net_input, num_outputs=num_mpi_planes, reuse=reuse)
            # Rescale alphas to (0, 1)
            alphas = (alphas + 1.) / 2.
            rgb = mpi_net_input[:, :, :, :3]
            for i in range(num_mpi_planes):
                curr_alpha = tf.expand_dims(alphas[:, :, :, i], -1)
                curr_rgba = tf.concat([rgb, curr_alpha], axis=3)
                if i == 0:
                    rgba_layers = curr_rgba
                else:
                    rgba_layers = tf.concat([rgba_layers, curr_rgba], axis=3)
            rgba_layers = tf.reshape(
                rgba_layers, [batch_size, img_height, img_width, num_mpi_planes, 4])

            return feat_lf, atts_lf, rgba_layers
        else:
            feat_lf, atts_lf, _ = self.mpi_net(mpi_net_input, num_outputs=num_mpi_planes, reuse=reuse)

            return feat_lf, atts_lf

    def mpi_render_view(self, rgba_layers, tgt_pose, planes):
        """Render a target view from an MPI representation.

        Args:
          rgba_layers: input MPI [batch, height, width, #planes, 4]
          tgt_pose: target pose to render from [batch, 3, 3]
          planes: list of depth for each plane
        Returns:
          rendered light field view [batch, height, width, 3]
        """
        batch_size, _, _ = tgt_pose.get_shape().as_list()
        depths = tf.constant(planes, shape=[len(planes), 1])
        depths = tf.tile(depths, [1, batch_size])
        rgba_layers = tf.transpose(rgba_layers, [3, 0, 1, 2, 4])
        proj_images = pj.projective_forward_translation(rgba_layers, tgt_pose, depths)
        proj_images_list = []
        for i in range(len(planes)):
            proj_images_list.append(proj_images[i])
        output_image = pj.over_composite(proj_images_list)
        return output_image

    def mpi_render_depth(self, rgba_layers, planes):
        """Render the depth map from an MPI representation.

        Args:
          rgba_layers: input MPI [batch, height, width, #planes, 4]
          planes: list of depth for each plane
        Returns:
          rendered depth map of center view [batch, height, width]
        """
        alpha_layers = rgba_layers[:, :, :, :, 3]
        alpha_layers = tf.transpose(alpha_layers, [3, 0, 1, 2])
        alpha_images_list = []
        for i in range(len(planes)):
            alpha_images_list.append(alpha_layers[i])
        for i in range(len(alpha_images_list)):
            alpha = alpha_images_list[i][:, :, :]
            curr_depth = planes[i]
            if i == 0:
                output = curr_depth
            else:
                depth_by_alpha = curr_depth * alpha
                output = depth_by_alpha + output * (1.0 - alpha)
        return output

    def format_network_input(self, psv_src_views, ref_pose, psv_src_poses, planes):
        """Format the network input (reference source view + PSV of other views).

        Args:
          psv_src_views: stack of source images (including the ref image)
                          [batch, height, width, 3*num_source]
          ref_pose: reference world-to-camera pose (where PSV is constructed)
                    [batch, 3, 3]
          psv_src_poses: input poses (world to camera) [batch, num_source-1, 3, 3]
          planes: list of scalar depth values for each plane
        Returns:
          net_input: [batch, height, width, (num_source-1)*#planes*3 + 3]
        """
        _, num_psv_source, _, _ = psv_src_poses.get_shape().as_list()
        ref_view = psv_src_views[:, :, :, :3]
        psv_src_views = psv_src_views[:, :, :, 3:]
        net_input = []
        net_input.append(ref_view)
        for i in range(num_psv_source):
            curr_pose = tf.matmul(psv_src_poses[:, i], tf.matrix_inverse(ref_pose))
            curr_image = psv_src_views[:, :, :, i * 3:(i + 1) * 3]
            curr_psv = pj.plane_sweep(curr_image, planes, curr_pose)
            net_input.append(curr_psv)
        net_input = tf.concat(net_input, axis=3)
        return net_input

    def inv_depths(self, start_depth, end_depth, num_depths):
        """Sample reversed, sorted inverse depths between a near and far plane.

        Args:
          start_depth: The first depth (i.e. near plane distance).
          end_depth: The last depth (i.e. far plane distance).
          num_depths: The total number of depths to create. start_depth and
              end_depth are always included and other depths are sampled
              between them uniformly according to inverse depth.
        Returns:
          The depths sorted in descending order (so furthest first). This order is
          useful for back to front compositing.
        """

        if (start_depth < 0.0) and (end_depth > 0.0):
            alpha = (0.0 - start_depth) / (end_depth - start_depth)
            depths = [start_depth, end_depth, 0.0]
            for i in range(1, round(alpha * num_depths)):
                fraction = float(i) / float(round(alpha * num_depths))
                depth = start_depth + (0.0 - start_depth) * fraction
                depths.append(depth)
            for i in range(1, int(num_depths - round(alpha * num_depths) - 1)):
                fraction = float(i) / float(int(num_depths - round(alpha * num_depths) - 1))
                depth = 0.0 + (end_depth - 0.0) * fraction
                depths.append(depth)
        else:
            depths = [start_depth, end_depth]
            for i in range(1, num_depths - 1):
                fraction = float(i) / float(num_depths - 1)
                depth = start_depth + (end_depth - start_depth) * fraction
                depths.append(depth)
        depths = sorted(depths)
        return depths[::-1]
