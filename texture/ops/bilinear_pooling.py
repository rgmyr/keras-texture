"""The bilinear pooling operation (see networks/bilinear_cnn.py)

TODO: - support for matrix square root described in "Improved Bilinear Pooling with CNNs"
        (claimed to add 2-3% accuracy on fine-grained benchmark datasets)
"""
import tensorflow as tf
from tensorflow.keras.layers import Flatten
#_all__

def bilinear_pooling(inputs):
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
    phi_I = tf.einsum('bijm,bijn->bmn', iA, iB)

    # sum --> avg (is this necessary?)
    #n_feat = tf.reduce_prod(tf.gather(tf.shape(iA), tf.constant([1,2])))
    #phi_I  = tf.divide(phi_I, tf.to_float(n_feat))

    # flatten
    phi_I = Flatten()(phi_I)

    # signed square root
    phi_I = tf.multiply(tf.sign(phi_I),tf.sqrt(tf.abs(phi_I)+1e-10))

    # L2 normalization
    z = tf.nn.l2_normalize(sgn_sqrt, axis=-1)

    return z
