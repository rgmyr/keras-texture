"""
Softmax weighted reduce_sum of multiple input operations (which should be stacked along first non-batch axis).

Reference:
    "DARTS: Differentiable Architecture Search", (Lui, Simonyan, Yang 2018)
    https://arxiv.org/abs/1806.09055
"""
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from tensorflow.keras.layers import Flatten


__all__ = ['DARTSEdge']


class DARTSEdge(tf.keras.layers.Layer):
    """
    Weighted elementwise addition of multiple input volumes or vectors.

    Input should be a list of operations, all with shape (batch, <ops_output_shape>).
    Output_shape is a single tensor (batch, <ops_output_shape>).
    """
    def __init__(self, **kwargs):
        super(DARTSEdge, self).__init__(**kwargs)


    def build(self, input_shape):
        self.num_ops = input_shape[1]
        self.trailing_axes = len(input_shape) - 2

        op_weights_shape = [1, self.num_ops] + (self.trailing_axes)*[1]

        self.op_weights = self.add_weight(name='op_weights',
                                       shape=op_weights_shape,
                                       initializer='ones',
                                       trainable=True)

        super(DARTSEdge, self).build(input_shape)


    def call(self, x):
        alpha = tf.nn.softmax(self.op_weights, axis=1)

        x_weighted = alpha * x

        return tf.reduce_sum(x_weighted, axis=1)


    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape.pop(1)

        return tf.TensorShape(output_shape)
