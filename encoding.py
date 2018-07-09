'''Keras 2.0+ / TensorFlow implementation of encoding layer proposed in:

@InProceedings{Zhang_2017_CVPR,
    author = {Zhang, Hang and Xue, Jia and Dana, Kristin},
    title = {Deep TEN: Texture Encoding Network},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {July},
    year = {2017}
}

'''
from keras import backend as K
from keras.engine.topology import Layer


class Encoding(Layer):
    '''Dictionary encoding layer with learnable codebook and residual weights.

    Encodes an collection of N features of size D into a KxD representation.
    
    If input.shape = (N, D) and n_codes = K, then
        codebook.shape = (K, D)
        residual.shape = (N, K, D)
        weights.shape  = (N, K)
        output.shape   = (K, D)

    Input of shape (batch, H, W, D) from e.g., a conv layer, should be 
    flattened to (batch, N=H*W, D) before using as Encoding input tensor.
    
    # Arguments:
        n_codes : number of codewords in the learnable codebook
    '''
    def __init__(self, n_codes, **kwargs):
        super(Encoding, self).__init__(**kwargs)
        self.n_codes = n_codes

    def build(self, input_shape):
        # TODO: opt to init codebook manually? Or with GMM?

        self.C = self.add_weights(name='codebook',
                                 shape=(self.n_codes, input_shape[-1]),
                                 initializer='orthogonal',
                                 trainable=True)

        self.S = self.add_weights(name='scale_factors',
                                 shape=(self.n_codes),
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
        return E

    def compute_output_shape(self, input_shape):
        return (self.n_codes, input_shape[-1])





