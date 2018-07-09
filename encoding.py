'''Keras 2.0+ / TensorFlow implementation of encoding layer proposed in:

Hang Zhang, Jia Xue, and Kristin Dana. "Deep TEN: Texture Encoding Network."
*The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2017*

'''
from keras import backend as K
from keras.engine.topology import Layer

def aggregate(W, X, C):
    '''
    Compute: E_k = Sum_i (W_ik*(X_i - C_k))
    '''
    pass

def scaledL2(X, C, S):
    '''
    Compute: W_ik = S_k * ||X_i - C_k||^2
    '''
    pass


class Encoding(Layer):
    '''Dictionary encoding layer with learnable codebook and residual weights.

    Encodes an collection of N features of size D into a KxD representation.
    
    If input.shape = (N, D) and num_codes = K, then
        codebook.shape = (K, D)
        residual.shape = (N, K, D)
        weights.shape  = (N, K)
        output.shape   = (K, D)

    Input of shape (batch, H, W, D) from e.g., a conv layer, should be 
    flattened to (batch, N=H*W, D) before using as Encoding input tensor.

    TODO: - allow either input_shape
          - support for dropout?
    
    # Arguments:
        D : size of the features in X (& codewords)
        K : number of codewords in the codebook
    '''
    def __init__(self, D, K, **kwargs):
        super(Encoding, self).__init__(**kwargs)
        self.D, self.K = D, K

    def build(self, input_shape):
        # TODO: opt to init codebook manually? Or with GMM?

        self.codewords = self.add_weights(name='codebook',
                                         shape=(self.K, self.D,),
                                         initializer='orthogonal', # should use uniform +/-std1?
                                         trainable=True)

        self.scale = self.add_weights(name='scale_factors',
                                     shape=(self.K,),
                                     initializer='uniform',
                                     trainable=True)

        super(Encoding, self).build(input_shape)

    def call(self, x):
        ''' 
        Pseudo-code
        -----------
        Residuals : r_ik = x_i - C_k
        Weights   : w_ij = exp(-S_j*||r_ij||^2) / Sum_k(exp(-S_k*||r_ik||^2)) 
        Aggregate : E_j  = Sum_i(w_ij * r_ij)
        '''
         
        
        # code-wise dot product
        #E = tf.einsum('ij,ij->j', W, R)

        W = K.softmax(scaledL2(C, self.codewords, self.scale))

        E = aggregate(W, X, self.codewords)

        return E

    def compute_output_shape(self, input_shape):
        return (self.K, self.D)





