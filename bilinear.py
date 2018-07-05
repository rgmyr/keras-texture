'''Utilities for creating Bilinear CNN models in Keras w/ TensorFlow backend as described in:

@inproceedings{lin2015bilinear,
    Author = {Tsung-Yu Lin, Aruni RoyChowdhury, and Subhransu Maji},
    Title = {Bilinear CNNs for Fine-grained Visual Recognition},
    Booktitle = {International Conference on Computer Vision (ICCV)},
    Year = {2015}
}

bilinear.pool(inputs) : bilinear (feature-wise outer product) average pooling
bilinear.combine(fA, fB, ...) : use bilinear.pool merge two models into single BCNN

TODO: support for matrix square root layer described in "Improved Bilinear Pooling with CNNs"
      (claimed to add 2-3% accuracy on fine-grained benchmark datasets)
'''
import tensorflow as tf
from keras import backend as K
from keras import models, layers


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

    # sum --> avg
    n_feat = tf.reduce_prod(tf.gather(tf.shape(iA), tf.constant([1,2])))
    phi_I  = tf.divide(phi_I, tf.to_float(n_feat))

    # signed square root
    y_ssqrt = tf.multiply(tf.sign(phi_I),tf.sqrt(tf.abs(phi_I)+1e-12))

    # L2 normalization
    z_L2 = tf.nn.l2_normalize(y_ssqrt, axis=1)

    return z_L2

 
def combine(fA, fB, input_shape, n_classes, n_fc_layers=0, n_fc_hidden=[None]):
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
    n_fc_layers : int, optional
        Number of Dense layers between bilinear pooling and softmax. Default=0.
    n_fc_hidden : iterable, optional
        Iterable of ints, specifying number of neurons in each extra Dense layer.


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

    for _, N in zip(range(n_fc_layers), n_fc_hidden):
        x = layers.Dense(N, activation='relu')(x)

    x = layers.Dense(n_classes, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=x)

    return model

