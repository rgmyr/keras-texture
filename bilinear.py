'''Utilities for creating Bilinear CNN models in Keras w/ TensorFlow backend as described in:

@inproceedings{lin2015bilinear,
    Author = {Tsung-Yu Lin, Aruni RoyChowdhury, and Subhransu Maji},
    Title = {Bilinear CNNs for Fine-grained Visual Recognition},
    Booktitle = {International Conference on Computer Vision (ICCV)},
    Year = {2015}
}

bilinear.BilinearModel: keras layer for weighted bilinear modeling of two 1-D feature vectors
bilinear.pooling(inputs): bilinear (feature-wise outer product) average pooling
bilinear.combine(fA, fB, ...): use bilinear.pooling to merge two models into single BCNN

TODO: - tests for BilinearModel layer
      - support for matrix square root layer described in "Improved Bilinear Pooling with CNNs"
        (claimed to add 2-3% accuracy on fine-grained benchmark datasets)
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


def pooling(inputs):
    '''Pool outer products of local features. Returns tf Function usable with keras.layers.Lambda.

    Parameters
    ----------
    inputs : (tf.Tensor, tf.Tensor)
        Both tensors should be 4D (channels last), with same shape in all but channels dimension.
        
    Returns
    -------
    phi_I : tensorflow Function
        Symbolic function encapsulating pooling and normalization operations.
    '''
    iA, iB = inputs

    # sum pooling outer product
    phi_I = tf.einsum('ijkm,ijkn->imn', iA, iB)

    # sum --> avg (is this necessary?)
    n_feat = tf.reduce_prod(tf.gather(tf.shape(iA), tf.constant([1,2])))
    phi_I  = tf.divide(phi_I, tf.to_float(n_feat))

    # signed square root
    y_ssqrt = tf.multiply(tf.sign(phi_I),tf.sqrt(tf.abs(phi_I)+1e-12))

    # L2 normalization
    z_L2 = tf.nn.l2_normalize(y_ssqrt, axis=1)

    return z_L2

 
def combine(fA, fB, input_shape, n_classes, fc_layers=[]):
    '''Combine two feature extracting CNNs into single Model with bilinear_pooling + FC layers.
       fA and fB should output 4D tensors of equal shape, except (optionally) in # of channels.

    Parameters
    ----------
    fA : keras.models.Model
        Feature network A. Should output features (N, H, W, cA).
    fB : keras.models.Model or `None`
        Feature network B. Should output features (N, H, W, cB).
        If `None`, will return symmetric BCNN using fA.
    input_shape : tuple of int
        Shape of input images. Must be compatible with fA.input & fB.input.
    n_classes : int
        Number of classes for softmax layer
    fc_layers : iterable of int, optional
        Sizes for additional Dense layers between bilinear vector and softmax. Default=[].

    Returns
    -------
    BCNN : keras.models.Model
        Single bilinear CNN composed from fA & fB (asymmetric) or fA with itself (symmetric)
    '''
    input_layer = layers.Input(shape=input_shape)

    outA = fA(input_layer)
    if fB is None:
        outB = outA
    else:
        outB = fB(input_layer)

    x = layers.Lambda(pooling, name='bilinear_pooling')([outA, outB])
    x = layers.Flatten()(x)

    for N in fc_layers:
        x = layers.Dense(N, activation='relu')(x)

    x = layers.Dense(n_classes, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=x)

    return model


def add_1x1_conv(f, n_channels, use_bias=False):
    '''Map output of a model to n_channels with a 1x1 convolution.
    
    Parameters
    ----------
    f : keras.models.Model
        Feature network with output shape (N, H, W, C).
    n_channels : int
        Number of output channels for 1x1 conv.
        Usually < C, but can be any int value.
    
    Returns
    -------
    f_n : keras.models.Model
        Feature network f + 1x1 conv. Output shape now (N, H, W, n_channels).
    '''
    x = f.output

    x = layers.Conv2D(n_channels, (1,1),
                      kernel_initializer='orthogonal',
                      use_bias=use_bias)(x)

    return models.Model(inputs=f.input, outputs=x)

