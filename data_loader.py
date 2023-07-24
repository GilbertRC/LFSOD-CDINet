"""Class definition of the data loader.
"""
from __future__ import division

import os.path
import tensorflow as tf
from utils import build_matrix


class TrainLoader(object):
    """Loader for training data."""

    def __init__(self,
                 view_root,
                 gt_root,
                 batch_size,
                 train_height,
                 train_width,
                 num_source=5):
        self.batch_size = batch_size
        self.train_height = train_height
        self.train_width = train_width
        self.num_source = num_source

        view_1_root = view_root + 'view_1(center)/'
        view_2_root = view_root + 'view_2(lefttop)/'
        view_3_root = view_root + 'view_3(rightbottom)/'
        view_4_root = view_root + 'view_4(righttop)/'
        view_5_root = view_root + 'view_5(leftbottom)/'
        gt_pose_root = view_root + 'gt_poses/'
        gt_view_root = view_root + 'gt_views/'
        assert tf.gfile.IsDirectory(view_1_root)  # Ensure the provided path is valid.
        assert len(tf.gfile.ListDirectory(view_1_root)) > 0  # Ensure that some data exists.
        assert tf.gfile.IsDirectory(view_2_root)  # Ensure the provided path is valid.
        assert len(tf.gfile.ListDirectory(view_2_root)) > 0  # Ensure that some data exists.
        assert tf.gfile.IsDirectory(view_3_root)  # Ensure the provided path is valid.
        assert len(tf.gfile.ListDirectory(view_3_root)) > 0  # Ensure that some data exists.
        assert tf.gfile.IsDirectory(view_4_root)  # Ensure the provided path is valid.
        assert len(tf.gfile.ListDirectory(view_4_root)) > 0  # Ensure that some data exists.
        assert tf.gfile.IsDirectory(view_5_root)  # Ensure the provided path is valid.
        assert len(tf.gfile.ListDirectory(view_5_root)) > 0  # Ensure that some data exists.
        assert tf.gfile.IsDirectory(gt_root)  # Ensure the provided path is valid.
        assert len(tf.gfile.ListDirectory(gt_root)) > 0  # Ensure that some data exists.
        assert tf.gfile.IsDirectory(gt_pose_root)  # Ensure the provided path is valid.
        assert len(tf.gfile.ListDirectory(gt_pose_root)) > 0  # Ensure that some data exists.
        assert tf.gfile.IsDirectory(gt_view_root)  # Ensure the provided path is valid.
        assert len(tf.gfile.ListDirectory(gt_view_root)) > 0  # Ensure that some data exists.
        self.views_1 = [view_1_root + f for f in os.listdir(view_1_root) if f.endswith('.png')]
        self.views_2 = [view_2_root + f for f in os.listdir(view_2_root) if f.endswith('.png')]
        self.views_3 = [view_3_root + f for f in os.listdir(view_3_root) if f.endswith('.png')]
        self.views_4 = [view_4_root + f for f in os.listdir(view_4_root) if f.endswith('.png')]
        self.views_5 = [view_5_root + f for f in os.listdir(view_5_root) if f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.gt_poses = [gt_pose_root + f for f in os.listdir(gt_pose_root) if f.endswith('.txt')]
        self.gt_views = [gt_view_root + f for f in os.listdir(gt_view_root) if f.endswith('.png')]
        self.size = len(self.views_1)

        self.view_1_pose = build_matrix([[1.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0],
                                         [0.0, 0.0, 1.0]])
        self.view_2_pose = build_matrix([[1.0, 0.0, -2.0],
                                         [0.0, 1.0, -2.0],
                                         [0.0, 0.0, 1.0]])
        self.view_3_pose = build_matrix([[1.0, 0.0, 2.0],
                                         [0.0, 1.0, 2.0],
                                         [0.0, 0.0, 1.0]])
        self.view_4_pose = build_matrix([[1.0, 0.0, 2.0],
                                         [0.0, 1.0, -2.0],
                                         [0.0, 0.0, 1.0]])
        self.view_5_pose = build_matrix([[1.0, 0.0, -2.0],
                                         [0.0, 1.0, 2.0],
                                         [0.0, 0.0, 1.0]])

        self.datasets = self.create_from_flags()

    # Create a dataset configured with the flags specified at the top of train file.
    def create_from_flags(self):
        """Convenience function to return a dataset configured by flags."""
        dataset = tf.data.Dataset.from_tensor_slices((self.views_1, self.views_2, self.views_3,
                                                      self.views_4, self.views_5, self.gts,
                                                      self.gt_poses, self.gt_views))
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(self.size, count=-1))
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            map_func=self._parse_function, batch_size=self.batch_size,
            num_parallel_batches=self.batch_size * 2, drop_remainder=False))
        dataset = dataset.prefetch(buffer_size=self.batch_size)

        return dataset

    def _parse_function(self, view_1_name, view_2_name, view_3_name, view_4_name, view_5_name,
                        gt_name, gt_pose_name, gt_view_name):
        """Reads data file, including: views, gts, gt_poses, gt_views.
        """

        def load_rgb_image(filename):
            contents = tf.read_file(filename)
            img = tf.image.convert_image_dtype(
                tf.image.decode_image(contents), tf.float32)
            shape_before_resize = tf.shape(img)
            original_width = shape_before_resize[1]
            original_height = shape_before_resize[0]
            resized = tf.squeeze(
                tf.image.resize_area(tf.expand_dims(img, axis=0), [self.train_height, self.train_width]),
                axis=0)
            resized.set_shape([self.train_height, self.train_width, 3])  # RGB images have 3 channels.
            scale_factor_hori = tf.cast(tf.shape(resized)[1] / original_width, dtype=tf.float32)
            scale_factor_verti = tf.cast(tf.shape(resized)[0] / original_height, dtype=tf.float32)
            return resized, scale_factor_hori, scale_factor_verti

        def load_binary_image(filename):
            contents = tf.read_file(filename)
            img = tf.image.convert_image_dtype(
                tf.image.decode_image(contents), tf.float32)
            resized = tf.squeeze(
                tf.image.resize_area(tf.expand_dims(img, axis=0), [self.train_height, self.train_width]),
                axis=0)
            resized.set_shape([self.train_height, self.train_width, 1])  # Binary images have 1 channels.
            return resized

        def load_pose_with_factor(filename, factor_h, factor_v):
            lines = tf.contrib.data.get_single_element((
                tf.data.TextLineDataset(filename).filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), '#'))))
            record_defaults = ([[0.0]] * 2)
            data = tf.decode_csv(lines, record_defaults, field_delim=' ')
            pose = build_matrix([[1.0, 0.0, -data[0]*factor_h],
                                 [0.0, 1.0, -data[1]*factor_v],
                                 [0.0, 0.0, 1.0]])
            return pose

        def change_pose_with_factor(factor_h, factor_v):
            scale_factor_l = build_matrix([[factor_h, 0.0, 0.0],
                                           [0.0, factor_v, 0.0],
                                           [0.0, 0.0, 1.0]])
            scale_factor_r = build_matrix([[1 / tf.to_float(factor_h), 0.0, 0.0],
                                           [0.0, 1 / tf.to_float(factor_v), 0.0],
                                           [0.0, 0.0, 1.0]])
            pose_1 = self.view_1_pose
            pose_2 = tf.matmul(tf.matmul(scale_factor_l, self.view_2_pose), scale_factor_r)
            pose_3 = tf.matmul(tf.matmul(scale_factor_l, self.view_3_pose), scale_factor_r)
            pose_4 = tf.matmul(tf.matmul(scale_factor_l, self.view_4_pose), scale_factor_r)
            pose_5 = tf.matmul(tf.matmul(scale_factor_l, self.view_5_pose), scale_factor_r)
            return pose_1, pose_2, pose_3, pose_4, pose_5

        # view
        view_1, factor_hori, factor_verti = load_rgb_image(view_1_name)
        view_2, _, _ = load_rgb_image(view_2_name)
        view_3, _, _ = load_rgb_image(view_3_name)
        view_4, _, _ = load_rgb_image(view_4_name)
        view_5, _, _ = load_rgb_image(view_5_name)
        # change pose with factor
        src_poses = change_pose_with_factor(factor_hori, factor_verti)
        # gt
        gt = load_binary_image(gt_name)
        gt_pose = load_pose_with_factor(gt_pose_name, factor_hori, factor_verti)
        gt_view, _, _ = load_rgb_image(gt_view_name)

        return view_1, view_2, view_3, view_4, view_5, src_poses, gt, gt_pose, gt_view

    def set_shapes(self, examples):
        """Set static shapes of the mini-batch of examples.

        Args:
          examples: a batch of examples
        Returns:
          examples with correct static shapes
        """
        b = self.batch_size
        h = self.train_height
        w = self.train_width
        s = self.num_source
        examples['ref_image'].set_shape([b, h, w, 3])
        examples['src_views'].set_shape([b, h, w, 3 * s])
        examples['ref_pose'].set_shape([b, 3, 3])
        examples['src_poses'].set_shape([b, s, 3, 3])
        examples['gt'].set_shape([b, h, w, 1])
        examples['gt_pose'].set_shape([b, 3, 3])
        examples['gt_view'].set_shape([b, h, w, 3])
        return examples

    def sample_batch(self):
        """Samples a batch of examples for training / testing.

        Returns:
          A batch of examples.
        """
        example = self.datasets.map(self.format_for_saliency)
        iterator = example.make_one_shot_iterator()
        return self.set_shapes(iterator.get_next())

    def format_for_saliency(self, view_1, view_2, view_3, view_4, view_5, src_poses, gt, gt_pose, gt_view):
        """Format the sampled sequence for CPD-MPI training/inference.
        """

        def normalize(img, mean, std):
            """Normalize the image for vgg16 input.
            """
            img_norm = (img - mean) / std
            return img_norm

        def preprocess_view(view):
            """Preprocess the view for CNN input with range [-1, 1].
             """
            return view * 2 - 1

        ref_image = normalize(view_1, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize to vgg16 input
        ref_pose = src_poses[0]
        gt_view = preprocess_view(gt_view)

        src_views = preprocess_view(tf.concat([view_1, view_2, view_3, view_4, view_5], axis=-1))
        src_poses = tf.stack([src_poses[0], src_poses[1], src_poses[2], src_poses[3], src_poses[4]], axis=1)

        # Put everything into a dictionary.
        instance = {}
        instance['ref_image'] = ref_image
        instance['src_views'] = src_views
        instance['ref_pose'] = ref_pose
        instance['src_poses'] = src_poses
        instance['gt'] = gt
        instance['gt_pose'] = gt_pose
        instance['gt_view'] = gt_view
        return instance


class TestLoader(object):
    """Loader for data."""

    def __init__(self,
                 view_root,
                 gt_root,
                 batch_size,
                 test_height,
                 test_width,
                 num_source=5):
        self.batch_size = batch_size
        self.test_height = test_height
        self.test_width = test_width
        self.num_source = num_source

        image_root = view_root + 'test_images/'
        view_1_root = view_root + 'view_1(center)/'
        view_2_root = view_root + 'view_2(lefttop)/'
        view_3_root = view_root + 'view_3(rightbottom)/'
        view_4_root = view_root + 'view_4(righttop)/'
        view_5_root = view_root + 'view_5(leftbottom)/'
        assert tf.gfile.IsDirectory(image_root)  # Ensure the provided path is valid.
        assert len(tf.gfile.ListDirectory(image_root)) > 0  # Ensure that some data exists.
        assert tf.gfile.IsDirectory(view_1_root)  # Ensure the provided path is valid.
        assert len(tf.gfile.ListDirectory(view_1_root)) > 0  # Ensure that some data exists.
        assert tf.gfile.IsDirectory(view_2_root)  # Ensure the provided path is valid.
        assert len(tf.gfile.ListDirectory(view_2_root)) > 0  # Ensure that some data exists.
        assert tf.gfile.IsDirectory(view_3_root)  # Ensure the provided path is valid.
        assert len(tf.gfile.ListDirectory(view_3_root)) > 0  # Ensure that some data exists.
        assert tf.gfile.IsDirectory(view_4_root)  # Ensure the provided path is valid.
        assert len(tf.gfile.ListDirectory(view_4_root)) > 0  # Ensure that some data exists.
        assert tf.gfile.IsDirectory(view_5_root)  # Ensure the provided path is valid.
        assert len(tf.gfile.ListDirectory(view_5_root)) > 0  # Ensure that some data exists.
        assert tf.gfile.IsDirectory(gt_root)  # Ensure the provided path is valid.
        assert len(tf.gfile.ListDirectory(gt_root)) > 0  # Ensure that some data exists.
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.png') or f.endswith('.jpg')]
        self.views_1 = [view_1_root + f for f in os.listdir(view_1_root) if f.endswith('.png') or f.endswith('.jpg')]
        self.views_2 = [view_2_root + f for f in os.listdir(view_2_root) if f.endswith('.png') or f.endswith('.jpg')]
        self.views_3 = [view_3_root + f for f in os.listdir(view_3_root) if f.endswith('.png') or f.endswith('.jpg')]
        self.views_4 = [view_4_root + f for f in os.listdir(view_4_root) if f.endswith('.png') or f.endswith('.jpg')]
        self.views_5 = [view_5_root + f for f in os.listdir(view_5_root) if f.endswith('.png') or f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.jpg')]
        self.name = os.listdir(image_root)
        for i in range(len(self.images)):
            if self.name[i].endswith('.jpg'):
                self.name[i] = self.name[i].split('.jpg')[0] + '.png'
        self.size = len(self.images)

        self.view_1_pose = build_matrix([[1.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0],
                                         [0.0, 0.0, 1.0]])
        self.view_2_pose = build_matrix([[1.0, 0.0, -2.0],
                                         [0.0, 1.0, -2.0],
                                         [0.0, 0.0, 1.0]])
        self.view_3_pose = build_matrix([[1.0, 0.0, 2.0],
                                         [0.0, 1.0, 2.0],
                                         [0.0, 0.0, 1.0]])
        self.view_4_pose = build_matrix([[1.0, 0.0, 2.0],
                                         [0.0, 1.0, -2.0],
                                         [0.0, 0.0, 1.0]])
        self.view_5_pose = build_matrix([[1.0, 0.0, -2.0],
                                         [0.0, 1.0, 2.0],
                                         [0.0, 0.0, 1.0]])

        self.datasets = self.create_from_flags()

    # Create a dataset configured with the flags specified at the top of train file.
    def create_from_flags(self):
        """Convenience function to return a dataset configured by flags."""
        dataset = tf.data.Dataset.from_tensor_slices((self.images, self.views_1, self.views_2,
                                                      self.views_3, self.views_4, self.views_5,
                                                      self.gts, self.name))
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            map_func=self._parse_function, batch_size=self.batch_size,
            num_parallel_batches=self.batch_size * 2, drop_remainder=False))
        dataset = dataset.prefetch(buffer_size=self.batch_size)

        return dataset

    def _parse_function(self, image_name, view_1_name, view_2_name, view_3_name,
                        view_4_name, view_5_name, gt_name, name):
        """Reads data file, including: views, gts.
        """

        def load_rgb_image(filename):
            contents = tf.read_file(filename)
            img = tf.image.convert_image_dtype(
                tf.image.decode_image(contents), tf.float32)
            shape_before_resize = tf.shape(img)
            original_width = shape_before_resize[1]
            original_height = shape_before_resize[0]
            resized = tf.squeeze(
                tf.image.resize_area(tf.expand_dims(img, axis=0), [self.test_height, self.test_width]),
                axis=0)
            resized.set_shape([self.test_height, self.test_width, 3])  # RGB images have 3 channels.
            scale_factor_hori = tf.cast(tf.shape(resized)[1] / original_width, dtype=tf.float32)
            scale_factor_verti = tf.cast(tf.shape(resized)[0] / original_height, dtype=tf.float32)
            return resized, scale_factor_hori, scale_factor_verti

        def load_binary_image(filename):
            contents = tf.read_file(filename)
            img = tf.image.convert_image_dtype(
                tf.image.decode_image(contents), tf.float32)
            return img

        def change_pose_with_factor(factor_h, factor_v):
            scale_factor_l = build_matrix([[factor_h, 0.0, 0.0],
                                           [0.0, factor_v, 0.0],
                                           [0.0, 0.0, 1.0]])
            scale_factor_r = build_matrix([[1 / tf.to_float(factor_h), 0.0, 0.0],
                                           [0.0, 1 / tf.to_float(factor_v), 0.0],
                                           [0.0, 0.0, 1.0]])
            pose_1 = self.view_1_pose
            pose_2 = tf.matmul(tf.matmul(scale_factor_l, self.view_2_pose), scale_factor_r)
            pose_3 = tf.matmul(tf.matmul(scale_factor_l, self.view_3_pose), scale_factor_r)
            pose_4 = tf.matmul(tf.matmul(scale_factor_l, self.view_4_pose), scale_factor_r)
            pose_5 = tf.matmul(tf.matmul(scale_factor_l, self.view_5_pose), scale_factor_r)
            return pose_1, pose_2, pose_3, pose_4, pose_5

        # image
        image, _, _ = load_rgb_image(image_name)
        # view
        view_1, factor_hori, factor_verti = load_rgb_image(view_1_name)
        view_2, _, _ = load_rgb_image(view_2_name)
        view_3, _, _ = load_rgb_image(view_3_name)
        view_4, _, _ = load_rgb_image(view_4_name)
        view_5, _, _ = load_rgb_image(view_5_name)
        # change pose with factor
        src_poses = change_pose_with_factor(factor_hori, factor_verti)
        # gt
        gt = load_binary_image(gt_name)

        return image, view_1, view_2, view_3, view_4, view_5, src_poses, gt, name

    def set_shapes(self, examples):
        """Set static shapes of the mini-batch of examples.

        Args:
          examples: a batch of examples
        Returns:
          examples with correct static shapes
        """
        b = self.batch_size
        h = self.test_height
        w = self.test_width
        s = self.num_source
        examples['ref_image'].set_shape([b, h, w, 3])
        examples['src_views'].set_shape([b, h, w, 3 * s])
        examples['ref_pose'].set_shape([b, 3, 3])
        examples['src_poses'].set_shape([b, s, 3, 3])
        # examples['gt'].set_shape([b, h, w, 1])
        return examples

    def sample_batch(self):
        """Samples a batch of examples for training / testing.

        Returns:
          A batch of examples.
        """
        example = self.datasets.map(self.format_for_saliency)
        iterator = example.make_one_shot_iterator()
        return self.set_shapes(iterator.get_next())

    def format_for_saliency(self, image, view_1, view_2, view_3, view_4, view_5, src_poses, gt, name):
        """Format the sampled sequence for MPI-CPD training/inference.
        """

        ref_image = image
        ref_pose = src_poses[0]

        src_views = tf.concat([view_1, view_2, view_3, view_4, view_5], axis=-1)
        src_poses = tf.stack([src_poses[0], src_poses[1], src_poses[2], src_poses[3], src_poses[4]], axis=1)

        # Put everything into a dictionary.
        instance = {}
        instance['ref_image'] = ref_image
        instance['src_views'] = src_views
        instance['ref_pose'] = ref_pose
        instance['src_poses'] = src_poses
        instance['gt'] = gt
        instance['name'] = name
        return instance
