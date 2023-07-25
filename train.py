"""Main script for training CDINet.
"""
from __future__ import division
import os
import time
import math
import tensorflow as tf
import logging

from data_loader import TrainLoader
from VGG_models import VGG
from MPI_models import MPI
from Fuse_module import CDIM
from utils import clip_gradient
from Fuse_module import upsample

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

flags = tf.app.flags
flags.DEFINE_boolean('continue_train', False,
                     'Continue training from previous checkpoint.')
flags.DEFINE_string('checkpoint_dir', 'checkpoints',
                    'Location to save the models.')
flags.DEFINE_string('experiment_name', 'model_HFUT',
                    'Name for the experiment to run.')
flags.DEFINE_integer('epoches', 40, 'Maximum number of training epoches.')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')
flags.DEFINE_integer('batch_size', 4, 'The size of a sample batch.')
flags.DEFINE_integer('train_height', 352, 'Image size of training dataset.')
flags.DEFINE_integer('train_width', 352, 'Image size of training dataset.')
flags.DEFINE_float('clip', 0.5, 'Gradient clipping margin')
flags.DEFINE_float('decay_rate', 0.1, 'Decay rate of learning rate')
flags.DEFINE_integer('decay_epoch', 30, 'Every n epochs decay learning rate')
flags.DEFINE_integer('random_seed', 8964, 'Random seed.')
flags.DEFINE_float('min_depth', -1.5, 'Minimum scene depth.')
flags.DEFINE_float('max_depth', 1.5, 'Maximum scene depth.')
flags.DEFINE_integer('num_planes', 16, 'Number of planes for plane sweep volume (PSV).')
FLAGS = flags.FLAGS


def min_max_norm(in_):
    max_ = tf.reduce_max(tf.reduce_max(in_, axis=2, keepdims=True), axis=1, keepdims=True)
    min_ = tf.reduce_min(tf.reduce_min(in_, axis=2, keepdims=True), axis=1, keepdims=True)
    in_ = in_ - min_
    return in_ / (max_ - min_ + 1e-8)


def deprocess_image(image):
    """Undo the preprocessing.

    Args:
        image: the input image in float with range [-1, 1]
    Returns:
        A new image converted to uint8 [0, 255]
    """
    image = (image + 1.) / 2.
    return tf.image.convert_image_dtype(image, dtype=tf.uint8)


def deprocess_depth(depth, planes):
    """Undo the preprocessing for the depth map.

    Args:
        depth: the input image in float with range [min(planes), max(planes)]
        planes: list of depth for each plane
    Returns:
        A new depth converted to uint8 [0, 255]
    """
    depth = 1.0 - (depth - min(planes)) / (max(planes) - min(planes))
    return tf.image.convert_image_dtype(depth, dtype=tf.uint8)


def train(data_loader, rgb_model, lf_model, fuse_module, continue_train=False):
    total_step = math.ceil(data_loader.size / FLAGS.batch_size)
    with tf.name_scope('build_train_graph'):
        with tf.name_scope('input_data'):
            train_batch = data_loader.sample_batch()
            psv_planes = lf_model.inv_depths(FLAGS.min_depth, FLAGS.max_depth, FLAGS.num_planes)

            ref_image = train_batch['ref_image']  # for 2D saliency detection
            src_views = train_batch['src_views']
            ref_pose = train_batch['ref_pose']
            src_poses = train_batch['src_poses']
            gt_pose = train_batch['gt_pose']
            gt_view = train_batch['gt_view']
            gts = train_batch['gt']

        with tf.name_scope('light_field'):
            mpi_net_input = lf_model.format_network_input(src_views, ref_pose, src_poses[:, 1:],
                                                          psv_planes)

            feat_lf, atts_lf, rgba_layers = lf_model.infer_mpi(mpi_net_input,
                                                               num_mpi_planes=FLAGS.num_planes,
                                                               is_render=True)
            output_view = lf_model.mpi_render_view(rgba_layers, gt_pose, psv_planes)
            output_depth = tf.expand_dims(lf_model.mpi_render_depth(rgba_layers, psv_planes), -1)

        with tf.name_scope('rgb'):
            feat_rgb, atts_rgb = rgb_model.vgg_net(ref_image)

        with tf.name_scope('fuse'):
            feat1, feat2, feat3, atts = fuse_module.fuse_module(atts_rgb, atts_lf, feat_rgb, feat_lf)
            dets = fuse_module.decoder_module(feat1, feat2, feat3)
            atts_rgb = upsample(atts_rgb)
            atts_lf = upsample(atts_lf)

        with tf.name_scope('loss'):
            loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=gts, logits=atts_rgb))
            loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=gts, logits=atts_lf))
            loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=gts, logits=dets))
            view_loss = tf.reduce_mean(tf.abs(output_view - gt_view))
            loss = loss1 + loss2 + loss3 + view_loss

        with tf.name_scope('train_op'):
            train_vars = [var for var in tf.trainable_variables()]
            global_step = tf.Variable(0, name='global_step', trainable=False)
            learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                                       decay_steps=total_step * FLAGS.decay_epoch,
                                                       decay_rate=FLAGS.decay_rate,
                                                       staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(loss, var_list=train_vars)
            grads_and_vars = clip_gradient(grads_and_vars, FLAGS.clip)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Summaries
    tf.summary.scalar('loss1_rgb', loss1)
    tf.summary.scalar('loss2_lf', loss2)
    tf.summary.scalar('loss3_dets', loss3)
    tf.summary.scalar('view_loss', view_loss)
    tf.summary.scalar('loss', loss)
    tf.summary.image('image', ref_image, 4)  # Input image
    tf.summary.image('output_atts_rgb', min_max_norm(tf.nn.sigmoid(atts_rgb)), 4)  # Output attention map of rgb
    tf.summary.image('output_atts_lf', min_max_norm(tf.nn.sigmoid(atts_lf)), 4)  # Output attention map of light field
    tf.summary.image('output_atts', atts, 4)  # Output attention map
    tf.summary.image('output_dets', min_max_norm(tf.nn.sigmoid(dets)), 4)  # Output detection map
    tf.summary.image('gt', gts, 4)  # Ground truth
    tf.summary.image('output_view', deprocess_image(output_view), 4)  # Output view
    tf.summary.image('output_depth', deprocess_depth(output_depth, psv_planes), 4)  # Output depth
    tf.summary.image('gt_view', deprocess_image(gt_view), 4)  # GT view

    with tf.name_scope('train'):
        parameter_count = tf.reduce_sum(
            [tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
        saver = tf.train.Saver(
            [var for var in tf.model_variables()] + [global_step], max_to_keep=100)
        sv = tf.train.Supervisor(
            logdir=FLAGS.checkpoint_dir, save_summaries_secs=0, saver=None)

        with sv.managed_session() as sess:
            logging.info('Trainable variables: ')
            for var in tf.trainable_variables():
                logging.info(var.name)
            logging.info('parameter_count = %d' % sess.run(parameter_count))
            if continue_train:
                checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
                if checkpoint is not None:
                    logging.info('Resume training from previous checkpoint')
                    saver.restore(sess, checkpoint)
            for epoch in range(1, FLAGS.epoches + 1):
                for step in range(1, total_step + 1):
                    start_time = time.time()
                    fetches = {'train': train_op,
                               'global_step': global_step,
                               'loss1': loss1,
                               'loss3': loss3}
                    if step % 20 == 0 or step == total_step:
                        fetches['summary'] = sv.summary_op

                    results = sess.run(fetches)
                    gs = results['global_step']

                    if step % 20 == 0 or step == total_step:
                        sv.summary_writer.add_summary(results['summary'], gs)
                        logging.info(
                            '[Epoch %.3d/%.3d][Step %.4d/%.4d] time: %4.4f/it, Loss1: %4.4f, Loss3: %4.4f' %
                            (epoch, FLAGS.epoches, step, total_step, time.time() - start_time,
                             results['loss1'], results['loss3']))

                if epoch % 5 == 0:
                    logging.info(' [*] Saving checkpoint to %s...' % FLAGS.checkpoint_dir)
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'model.latest'), epoch)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.set_random_seed(FLAGS.random_seed)
    FLAGS.checkpoint_dir += '/%s/' % FLAGS.experiment_name
    if not tf.gfile.IsDirectory(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

    RGB_model = VGG()
    LF_model = MPI()
    Fuse_module = CDIM()

    # Set up data
    view_root = './dataset/TrainingSet(HFUT)/'
    gt_root = './dataset/TrainingSet(HFUT)/train_gts_aug/'
    train_loader = TrainLoader(view_root, gt_root, FLAGS.batch_size, FLAGS.train_height, FLAGS.train_width)

    train(train_loader, RGB_model, LF_model, Fuse_module, FLAGS.continue_train)


if __name__ == '__main__':
    tf.app.run()
