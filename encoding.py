'''
Keras 2.0+ / TensorFlow implementation of learnable encoding layer, as proposed in:

    Hang Zhang, Jia Xue, and Kristin Dana. "Deep TEN: Texture Encoding Network."
    *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2017*

'''
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer


def scaledL2(R, S): 
    ''' L2 norm over features of R, scaled by a codeword-length vector S.
    Args:
        R (Tensor): 3-D tensor of shape (N,K,D)
        S (Tensor): 1-D tensor of shape (K,)

    Returns:
        Tensor: L2 norm of R along D, scaled over K axis by S
                Shape of output is (N,K)
    '''
    l2_R = tf.linalg.norm(R, axis=-1)
    return tf.multiply(l2_R, S)


class Encoding(Layer):
    '''Residual encoding layer with learnable codebook and assignment weights.

    Encodes an collection of NxD features into a KxD representation

    Input of shape (None, H, W, D) from e.g., a conv layer, should be 
    squeezed to (None, H*W, D) before feeding to Encoding layer.

    Args:
        D (int): dimensionality of features (and codewords)
        K (int): number of codewords to learn

    TODO: - support for dropout
          - support for (H, W, D) input
          - better initialization params
    '''
    def __init__(self, D, K, **kwargs):
        super(Encoding, self).__init__(**kwargs)
        self.D, self.K = D, K

    def build(self, input_shape):
        # TODO: implement std computation + uniform sampling

        self.codes = self.add_weight(name='codebook',
                                    shape=(self.K, self.D,),
                                    initializer='orthogonal', # should use uniform +/-std1?
                                    trainable=True)

        self.scale = self.add_weight(name='scale_factors',
                                    shape=(self.K,),
                                    initializer='uniform',
                                    trainable=True)

        super(Encoding, self).build(input_shape)

    def call(self, x):
        ndim = K.ndim(x)

        # TODO: make this work (tuple error?)
        #if ndim == 4:
        #    dims = K.shape(x)
        #    x = K.reshape(x, (-1, dims[1]*dims[2], self.D))

        if ndim != 3:
            raise ValueError('`x` should have shape BxNxD')
        
        # Residual vectors
        N = x.shape[1]
        _x_i = K.repeat_elements(x, self.K, 1)
        _c_k = K.tile(self.codes, (N, 1))
        R = K.reshape(_x_i - _c_k, (N, self.K, self.D)) 
        
        # Assignment weights
        W_ik = K.softmax(scaledL2(R, self.scale))

        # Aggregation
        E = tf.einsum('ik,ikd->kd', W_ik, R)

        return E

    def compute_output_shape(self, input_shape):
        return (self.K, self.D)


