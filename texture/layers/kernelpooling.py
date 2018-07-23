'''Implementation of polynomial Kernel Pooling layer with learnable composition: 
@conference{cui2017cvpr,
    title = {Kernel Pooling for Convolutional Neural Networks},
    author = {Yin Cui and Feng Zhou and Jiang Wang and Xiao Liu and Yuanqing Lin and Serge Belongie},
    url = {https://vision.cornell.edu/se3/wp-content/uploads/2017/04/cui2017cvpr.pdf},
    year = {2017},
    booktitle = {Computer Vision and Pattern Recognition (CVPR)},
}
_generate_sketch_matrix() borrowed from: https://github.com/ronghanghu/tensorflow_compact_bilinear_pooling
sequential_batch_[i]ff from the same repo would be useful for avoiding OOM errors w/ arbitrary batch size
    - does source need an update?
'''
import numpy as np
import tensorflow as tf

from math import factorial

from keras import backend as K
from keras.engine.topology import Layer


__all__ = ['AlphaInitializer', 'KernelPooling']


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
    Return a sparse matrix used for tensor/count sketch operation,
    which is random feature projection from input_dim-->output_dim.
    
    Parameters
    ----------
    rand_h: array, shape=(input_dim,)
        Vector containing indices in interval `[0, output_dim)`. 
    rand_s: array, shape=(input_dim,)
        Vector containing values of 1 and -1.
    output_dim: int
        The dimensions of the count sketch vector representation.
    Returns
    -------
    sparse_sketch_matrix : SparseTensor
        A sparse matrix of shape [input_dim, output_dim] for count sketch.
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


def _estimate_gamma(X_train):
    '''Estimate gamma for RBF approximation.'''
    assert isinstance(X_train, np.ndarray), 'X_train must be a numpy array of feature vectors'
    assert X_train.ndim in [3,4], 'X_train must be a 3D or 4D array of shape (batch,...,C)'
    if X_train.ndim == 4:
        X_train.reshape(X_train.shape[0], -1, X_train.shape[-1])
    
    # compute mean intra-image inner product
    pair_count = 0
    inner_sum = 0
    for i in range(X_train.shape[0]):
        dots = X_train[i].dot(X_train[i].T)
        pair_count += dots.size
        inner_sum += dots.sum()

    # return the reciprocal
    return pair_count / inner_sum


class AlphaInitializer():
    '''Callable for setting initial composition_weights given `gamma`. 
    Following the Taylor series expansion of the RBF kernel:
        K_RBF(x, y) = Sum_i(beta * \frac{(2*gamma)^i}{i!})
    We assume that input vectors are L2-normalized, in which case we have:
        (alpha_i)^2 = exp(-2*gamma)*\frac{(2*gamma)^i}{i!}
    '''

    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, shape, dtype=float):
        assert len(shape)==1, 'Only use AlphaInitializer on 1D weights'
        gam2 = np.array([2*self.gamma]*shape[0])
        beta = np.exp(-gam2)
        numer = np.power(gam2, range(gam2.size))
        denom = np.array([factorial(i) for i in range(gam2.size)])
        alpha = np.sqrt(beta * numer / denom)
        return K.variable(alpha)


class KernelPooling(Layer):
    '''Kernel Pooling layer with learnable composition weights. Takes convolution output volume as input, 
    outputs a Taylor series kernel of order `p`. By default the weights are initialized to approximate
    the Gaussian RBF kernel. See the paper for more detailed exposition. Kernel vectors are average
    pooled over all spatial locations (h_i, w_j).

    `output_shape=(batches,1+C+Sum(d_2,...,d_p))`, for `input_shape=(batches,H,W,C)`. This implementation 
    follows the paper in assuming that `d_i=d_bar` for all `i>=2`, so `d=1+C+(p-1)*d_i`.

    Parameters
    ----------
    p : int, optional
        Order of the polynomial approximation, default=3
    d_i : int, optional
        Dimensionality of output features of each order {2,...,p}, default=4096.
    seed : int, optional
        Random seed for generating sketch matrices. 
        If given, will use range(seed, seed+p) for `h`, and range(seed+p, seed+2p) for `s`.
    gamma : float, optional
        RBF kernel approximation parameter, default=1e-4.
    X_train : array, optional
        Training set of features from which to estimate gamma s.t. kernel closely approximates RBF. 
        If provided, will use reciprocal of mean inner product values. Otherwise, default used.
    '''

    def __init__(self, p=3, d_i=4096, seed=0, gamma=1e-4, X_train=None, **kwargs):
        self.p = p
        self.d_i = d_i
        self.h_seed = seed
        self.s_seed = seed + p
        if X_train is not None:
            self.gamma = _estimate_gamma(X_train)
        else:
            self.gamma = gamma
        super(KernelPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        #self._shapecheck(input_shape)
        # Initialize composition weights, RBF approximation
        alpha_init = AlphaInitializer(self.gamma)
        self.alpha = self.add_weight(name='composition_weights',
                                    shape=(self.p+1,),
                                    initializer=alpha_init,
                                    trainable=True)

        # Generate sketch matrices, need `p` sets of {h_t, s_t}
        self.C = input_shape[-1]
        self.sketch_matrices = []
        h_seeds = range(self.h_seed, self.h_seed+self.p)
        s_seeds = range(self.s_seed, self.s_seed+self.p)
        for hs, ss in zip(h_seeds,s_seeds):
            np.random.seed(hs)
            h_t = np.random.randint(self.d_i, size=self.C)
            np.random.seed(ss)
            s_t = 2*np.random.randint(2, size=self.C)-1
            self.sketch_matrices.append(_generate_sketch_matrix(h_t, s_t, self.d_i))
            

    def call(self, x):
        assert K.ndim(x) == 4, 'Should only call KP layer on input_shape (batches,H,W,C)'
        input_dims = K.shape(x)
        
        # zeroth and first order terms
        zeroth = self.alpha[0]*K.ones((input_dims[0],1))
        first  = self.alpha[1]*tf.reduce_mean(x, axis=[1,2])

        # flatten to feature vectors
        x_flat = K.reshape(x, (-1, self.C))
        
        # Compute the Count Sketches C_t over feature vectors
        sketches = []
        for t in range(self.p):
            sketches.append(tf.transpose(tf.sparse_tensor_dense_matmul(self.sketch_matrices[t],
                                         x_flat, adjoint_a=True, adjoint_b=True)))
            
        # stack and reshape [(b*h*w, d_i)], len=p --> (b, h*w, p, d_i) 
        x_sketches = K.reshape(K.stack(sketches, axis=-2), (input_dims[0], -1, self.p, self.d_i))
        
        # Compute fft (operates on inner-most axis)
        x_fft = _fft(tf.complex(real=x_sketches, imag=K.zeros_like(x_sketches)), False, 128)
        
        # Cumulative product along order dimension, discard first order
        x_fft_cp = K.cumprod(x_fft, axis=-2)[:, :, 1:, :]
        
        # Inverse fft, avg pool over spatial locations
        x_ifft = tf.reduce_mean(tf.real(_ifft(x_fft_cp, False, 128)), axis=1)
        
        # Apply weights over orders p >= 2
        x_p = x_ifft*K.reshape(self.alpha[2:], (1,self.p-1,1))
        
        # Concatenate to full order-p kernel approximation vector
        phi_x = K.concatenate([zeroth, first, K.reshape(x_p, (input_dims[0],-1))])
        
        # Return the transformed + l2-normed kernel vector
        phi_x = tf.multiply(tf.sign(phi_x),tf.sqrt(tf.abs(phi_x)+1e-12))

        return tf.nn.l2_normalize(phi_x, axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1+input_shape[-1]+(self.p-1)*self.d_i)


