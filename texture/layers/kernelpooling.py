'''Implementation of generalized polynomial kernel pooling layer:

@conference{cui2017cvpr,
    title = {Kernel Pooling for Convolutional Neural Networks},
    author = {Yin Cui and Feng Zhou and Jiang Wang and Xiao Liu and Yuanqing Lin and Serge Belongie},
    url = {https://vision.cornell.edu/se3/wp-content/uploads/2017/04/cui2017cvpr.pdf},
    year = {2017},
    booktitle = {Computer Vision and Pattern Recognition (CVPR)},
}

_generate_sketch_matrix() borrowed from: https://github.com/ronghanghu/tensorflow_compact_bilinear_pooling

'''
import numpy as np
import tensorflow as tf


def _fft(bottom, sequential, compute_size):
    #if sequential:
    #    return sequential_batch_fft(bottom, compute_size)
    #else:
    return tf.fft(bottom)

def _ifft(bottom, sequential, compute_size):
    #if sequential:
    #    return sequential_batch_ifft(bottom, compute_size)
    #else:
    return tf.ifft(bottom)

def _generate_sketch_matrix(rand_h, rand_s, output_dim):
    """
    Return a sparse matrix used for tensor sketch operation in compact bilinear
    pooling
    Args:
        rand_h: an 1D numpy array containing indices in interval `[0, output_dim)`.
        rand_s: an 1D numpy array of 1 and -1, having the same shape as `rand_h`.
        output_dim: the output dimensions of compact bilinear pooling.
    Returns:
        a sparse matrix of shape [input_dim, output_dim] for tensor sketch.
    """

    # Generate a sparse matrix for tensor count sketch
    rand_h = rand_h.astype(np.int64)
    rand_s = rand_s.astype(np.float32)
    assert(rand_h.ndim==1 and rand_s.ndim==1 and len(rand_h)==len(rand_s))
    assert(np.all(rand_h >= 0) and np.all(rand_h < output_dim))

    input_dim = len(rand_h)
    indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],
                              rand_h[..., np.newaxis]), axis=1)
    sparse_sketch_matrix = tf.sparse_reorder(
        tf.SparseTensor(indices, rand_s, [input_dim, output_dim]))
    return sparse_sketch_matrix

