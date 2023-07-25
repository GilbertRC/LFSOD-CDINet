"""Main script for testing CDINet.
"""
from __future__ import division
import os
import time

import tensorflow as tf
import numpy as np

from data_loader import TestLoader
from model.VGG_models import VGG
from model.MPI_models import MPI
from model.Fuse_module import CDIM
from utils import upsample
from utils import write_image

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

flags = tf.app.flags
flags.DEFINE_string('model_name', 'model_HFUT',
                    'Name of the model to use for inference.')
flags.DEFINE_string('model_root', 'checkpoints/',
                    'Root directory for model checkpoints.')
flags.DEFINE_integer('batch_size', 1, 'The size of a sample batch.')
flags.DEFINE_float('min_depth', -1.5, 'Minimum scene depth.')
flags.DEFINE_float('max_depth', 1.5, 'Maximum scene depth.')
flags.DEFINE_integer('num_planes', 16, 'Number of planes for plane sweep volume (PSV).')
FLAGS = flags.FLAGS

dataset_path = './dataset/TestSet/'


def min_max_norm(in_):
    max_ = tf.reduce_max(tf.reduce_max(in_, axis=2, keepdims=True), axis=1, keepdims=True)
    min_ = tf.reduce_min(tf.reduce_min(in_, axis=2, keepdims=True), axis=1, keepdims=True)
    in_ = in_ - min_
    return in_ / (max_ - min_ + 1e-8)


def normalize(img, mean, std):
    """Normalize the image for vgg16 input.
    """
    img_norm = (img - mean) / std
    return img_norm


def preprocess_view(view):
    """Preprocess the view for MPI-CNN input with range [-1, 1].
     """
    return view * 2 - 1


def test(test_loader, rgb_model, lf_model, fuse_module, save_res):
    with tf.name_scope('input_data'):
        test_batch = test_loader.sample_batch()
        psv_planes = lf_model.inv_depths(FLAGS.min_depth, FLAGS.max_depth, FLAGS.num_planes)
        # normalize to vgg16 input
        ref_image = normalize(test_batch['ref_image'], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # preprocess view for MPI network
        src_views = preprocess_view(test_batch['src_views'])
        ref_pose = test_batch['ref_pose']
        src_poses = test_batch['src_poses']
        gts = test_batch['gt']
        names = test_batch['name']

    with tf.name_scope('light_field'):
        mpi_net_input = lf_model.format_network_input(src_views, ref_pose, src_poses[:, 1:],
                                                      psv_planes)

        feat_lf, atts_lf = lf_model.infer_mpi(mpi_net_input,
                                              num_mpi_planes=FLAGS.num_planes,
                                              is_render=False)

    with tf.name_scope('rgb'):
        feat_rgb, atts_rgb = rgb_model.vgg_net(ref_image)

    with tf.name_scope('fuse'):
        feat1, feat2, feat3, atts = fuse_module.fuse_module(atts_rgb, atts_lf, feat_rgb, feat_lf)
        dets = fuse_module.decoder_module(feat1, feat2, feat3)

    saver = tf.train.Saver([var for var in tf.model_variables()])
    ckpt_dir = os.path.join(FLAGS.model_root, FLAGS.model_name)
    ckpt_file = ckpt_dir + '\\model.latest-40'
    sv = tf.train.Supervisor(logdir=ckpt_dir, saver=None)
    config = tf.ConfigProto()

    config.gpu_options.allow_growth = True
    with sv.managed_session(config=config) as sess:
        tf.reset_default_graph()
        saver.restore(sess, ckpt_file)

        for i in range(test_loader.size):
            fetches = {'detection': dets,
                       'gt': gts,
                       'name': names}
            results = sess.run(fetches)

            detection = results['detection']
            gt = results['gt']
            im_name = results['name'][0].decode()

            detection = np.squeeze(detection)
            detection = upsample(detection, [gt.shape[2], gt.shape[1]])
            detection = 1 / (1 + np.exp(-detection))  # sigmoid
            detection = (detection - detection.min()) / (detection.max() - detection.min() + 1e-8)
            write_image(save_res + im_name, detection)


def main(_):
    test_height = 352
    test_width = 352

    RGB_model = VGG()
    LF_model = MPI()
    Fuse_module = CDIM()

    test_datasets = ['HFUT-Lytro Illum(TIP2020)', 'HFUT-Lytro(TOMCCAP2017)']
    # test_datasets = ['DUTLF-V2']
    for dataset in test_datasets:
        save_path = './results/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Set up data
        view_root = dataset_path + dataset + '/'
        gt_root = dataset_path + dataset + '/test_gts/'
        test_loader = TestLoader(view_root, gt_root, FLAGS.batch_size, test_height, test_width)

        test(test_loader, RGB_model, LF_model, Fuse_module, save_path)


if __name__ == '__main__':
    tf.app.run()
