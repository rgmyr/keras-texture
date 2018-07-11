''' Keras layer for weighted bilinear modeling of two 1-D feature vectors

TODO: - tests for BilinearModel layer
'''
import tensorflow as tf
from keras import backend as K
from keras import models, layers
from keras.engine.topology import Layer


def BilinearModel(Layer):
    '''Weighted bilinear model of two inputs. Useful for learning a model of linear interactions
    between seperate feature types (e.g., texture X spatial) or scales (e.g., dense X dilated).

    #TODO: finish docstring
    '''
    
    def __init__(self, **kwargs):
        super(BilinearModel, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape)!=2:
            raise ValueError('Input to `BilinearModel` layer should be '
                             'a list of exactly two inputs')
        # TODO: enforce 2-D (batches, N) (batches, M)

        self.shapeA, self.shapeB = input_shape[0][1], input_shape[1][1]
        
        self.weights = self.add_weight(name='outer_prod_weights',
                                       shape=(self.shapeA, self.shapeB),
                                       initializer='glorot_normal',
                                       trainable=True)
    def call(self, x):
        if not isinstance(x, list) or len(x)!=2:
            raise ValueError('A `BilinearModel` layer should be called '
                             'on a list of exactly two inputs')
        if K.int_shape(x[0])[1]!=self.A or K.int_shape(x[1])[1]!=self.B:
            raise ValueError('Unexpected `BilinearModel` input_shapes')

        weighted_outer = tf.multiply(self.W, tf.einsum('bi,bj->bij', x[0], x[1]))

        return K.Flatten(weighted_outer)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.shapeA, self.shapeB)

