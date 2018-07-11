'''
Keras 2.0+ / TensorFlow implementation of learnable encoding layer, as proposed in:

    Hang Zhang, Jia Xue, and Kristin Dana. "Deep TEN: Texture Encoding Network."
    *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2017*

'''
import tensorflow as tf
from keras import backend as K
from keras.initializers import RandomUniform
from keras.engine.topology import Layer


def scaledL2(R, S): 
    ''' L2 norm over features of R, scaled by a codeword-length vector S.
    Args:
        R (Tensor): 4-D tensor of shape (batches,N,K,D) 
                 or 3-D tensor of shape (N,K,D)
        S (Tensor): 1-D tensor of shape (K,)

    Returns:
        Tensor: L2 norm of R along D, scaled over K axis by S
                Shape of output is (batches,N,K) or (N,K)
    '''
    l2_R = tf.linalg.norm(R, axis=-1)
    return tf.multiply(l2_R, S)


class Encoding(Layer):
    '''Residual encoding layer with learnable codebook and scaling weights.

    Encodes an collection of NxD features into set of KxD vectors.

    Input of shape (batches, H, W, D) from e.g., a conv layer, should be 
    collapsed to (batches, H*W, D) before feeding to Encoding layer. This
    should not be necessary in a later release.

    Args:
        K (int): number of codewords to learn
        dropout (float): dropout rate between [0.0,1.0) (default=`None`)
        l2_normalize (bool): normalize output vectors (default=`True`)

    TODO: - support for dropout
          - support for (H, W, D) input
          - std computation + uniform initiliazation
    '''
    def __init__(self, K, dropout=None, l2_normalize=True, **kwargs):
        super(Encoding, self).__init__(**kwargs)
        self.K = K
        self.l2_normalize = l2_normalize
        self.dropout_rate = dropout


    def build(self, input_shape):
        self.D = input_shape[-1]

        std1 = 1./((self.K*self.D)**(0.5))
        init_codes = RandomUniform(-std1, std1)
        self.codes = self.add_weight(name='codebook',
                                    shape=(self.K, self.D,),
                                    initializer=init_codes,
                                    trainable=True)

        init_scale = RandomUniform(-1, 0)
        self.scale = self.add_weight(name='scale_factors',
                                    shape=(self.K,),
                                    initializer=init_scale,
                                    trainable=True)

        super(Encoding, self).build(input_shape)

    def call(self, x):
        # Input is a 3-D or 4-D Tensor
        ndim = K.ndim(x)
        if ndim == 4:
            dims = K.int_shape(x)
            x = K.reshape(x, (-1, dims[1]*dims[2], self.D))
        elif ndim != 3:
            raise ValueError('Encoding input should have shape BxNxD or BxHxWxD')
        
        # Residual vectors
        n = x.shape[1]
        _x_i = K.repeat_elements(x, self.K, 1)
        _c_k = K.tile(self.codes, (n, 1))
        R = K.reshape(_x_i - _c_k, (-1, n, self.K, self.D)) 

        # Assignment weights, optional dropout
        if self.dropout_rate is not None:
            W_ik =  K.softmax(scaledL2(R, K.dropout(self.scale, self.dropout_rate)))
        else:
            W_ik = K.softmax(scaledL2(R, self.scale))

        # Aggregation
        E = tf.einsum('bik,bikd->bkd', W_ik, R)

        # Normalize encoding vectors
        if self.l2_normalize:
            E = tf.nn.l2_normalize(E, axis=-1)

        return E

    def compute_output_shape(self, input_shape):
        # (batches, codeworks, dimensions)
        return (input_shape[0], self.K, input_shape[-1])
