import tensorflow as tf
import numpy as np
import PIL.Image as pil


def clip_gradient(grads_and_vars, grad_clip):
    for i, (grad, var) in enumerate(grads_and_vars):
        if grad is not None:
            grads_and_vars[i] = (tf.clip_by_value(grad, -grad_clip, grad_clip), var)

    return grads_and_vars


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    optimizer._lr *= decay


def write_image(filename, image):
    """Save image to disk."""
    byte_image = np.clip(image * 255.0, 0, 255).astype('uint8')
    image_pil = pil.fromarray(byte_image)
    with tf.gfile.GFile(filename, 'w') as fh:
        image_pil.save(fh)


def upsample(im, size):
    im = np.array(pil.fromarray(im).resize(size, resample=pil.BILINEAR))

    return im


def build_matrix(elements):
    """Stacks elements along two axes to make a tensor of matrices.

    Args:
      elements: [n, m] matrix of tensors, each with shape [...].

    Returns:
      [..., n, m] tensor of matrices, resulting from concatenating
        the individual tensors.
    """
    rows = [tf.stack(row_elements, axis=-1) for row_elements in elements]
    return tf.stack(rows, axis=-2)
