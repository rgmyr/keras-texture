''' Keras layer for weighted bilinear modeling of two 1-D feature vectors

TODO: - tests for BilinearModel layer
'''
import tensorflow as tf
from keras import backend as K
from keras import models, layers
from keras.engine.topology import Layer


__all__ = ['BilinearModel']


def BilinearModel(Layer):
    '''Weighted bilinear model of two inputs. Useful for learning a model of linear interactions
    between seperate feature types (e.g., texture X spatial) or scales (e.g., dense X dilated), etc.

    #TODO: finish docstring
    '''
    
    def __init__(self, **kwargs):
        super(BilinearModel, self).__init__(**kwargs)

    def build(self, input_shape):
        self._shapecheck(input_shape)

        self.shapeA, self.shapeB = input_shape[0][1], input_shape[1][1]
        
        self.weights = self.add_weight(name='outer_prod_weights',
                                       shape=(self.shapeA, self.shapeB),
                                       initializer='glorot_normal',
                                       trainable=True)
    def call(self, x):
        self._shapecheck(x)

        weighted_outer = tf.multiply(self.weights, tf.einsum('bi,bj->bij', x[0], x[1]))

        return K.Flatten(weighted_outer)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.shapeA, self.shapeB)

    def _shapecheck(x):
        # both x and input_shape should be a list of len=2
        if not isinstance(x, list) or len(x) != 2:
            raise ValueError('A `BilinearModel` layer should be called on a list of exactly two inputs')

        # if input_shape, check dimensionality
        if isinstance(x[0], tuple):
            assert len(x[0]) == 2 and len(x[1]) == 2, '`BilinearModel` input tensors must be 2-D'
        
        # if x, they should match shapes from build()
        elif K.is_keras_tensor(x[0]):
            assert K.ndim(x[0]) == 2 and K.ndim(x[1]) == 2, '`BilinearModel` input tensors must be 2-D'
            shapeA, shapeB = K.int_shape(x[0])[1], K.int_shape(x[1])[1]
            if shapeA != self.shapeA or shapeB != self.shapeB:
                raise ValueError('Unexpected `BilinearModel` input_shape')


