"""The bilinear pooling operation (see networks/bilinear_cnn.py)

TODO: - support for matrix square root described in "Improved Bilinear Pooling with CNNs"
        (claimed to add 2-3% accuracy on fine-grained benchmark datasets)
"""
import tensorflow as tf
from tensorflow.keras.layers import Flatten
#_all__

def l2_pooling(x, epsilon=1e-12):
    '''L2 global pooling operation.

    Parameters
    ----------
    x : Tensor
        Should be 4D (channels last)

    Returns
    -------
    x_l2 : tensorflow Function
        Result of l2 pooling over spatial axes (1,2)
    '''

    square_sum = tf.reduce_sum(tf.square(x), [1,2])
    x_l2 = tf.sqrt(tf.maximum(square_sum, epsilon))

    return x_l2
