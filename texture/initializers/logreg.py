##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## This file is copied from https://github.com/keras-team/keras-contrib
##
## This source file is licensed under the MIT-style license found in the
## LICENSE file in the root directory of the keras-contrib source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import absolute_import
import numpy as np
from keras import backend as K
from keras.initializers import *
from keras.initializers import _compute_fans
from sklearn.linear_model import LogisticRegression


class LogReg(Initializer):
    """
    Initializer that generates weights for Dense layers by fitting a
    logistic regression model to a sample of (input, output) pairs.
    One expected use is for fitting a fully connected output layer
    to a bilinear pooling of pre-trained CNN features. Pooling output 
    is typically very large (e.g., 512*512), so that good initialization 
    can make convergence much faster and easier to achieve.

    Data is expected to be large, so 'saga' solver is used. 

    Parameters
    ----------
    _X : np.array, shape (n_samples, n_features)
        Sample of input to the layer
    _y : np.array, shape (n_samples, n_classes)
        Corresponding sample of layer output (e.g., one-hot encoded `y`)
    input_model : keras.models.Model instance or `None`
        If an input model is provided, it is assumed that the sample inputs 
        to the layer will be input_model.predict(_X). Otherwise, it is assumed
        that _X can be directly interpreted as inputs to the layer. Default=`None`.
    penalty : str, 'l1' or 'l2', default='l2'
        Norm to use in the regularization penalty
    seed : int, default=None
        Used to seed the random generator.
    """

    def __init__(self, _X, _y, input_model=None, penalty='l2', seed=None):
        if _X is None or _y is None:
            raise ValueError('Must pass _X and _y to construct LogReg initializer')
        if input_model is not None:
            _X = input_model.predict(_X)
        self.penalty = penalty
        self.seed = seed
        self._fit(_X, _y)

    def __call__(self, shape, dtype=None):
        rank = len(shape)

        if self.seed is not None:
            np.random.seed(self.seed)

        fan_in, fan_out = _compute_fans(shape, K.image_data_format())
        variance = 2 / fan_in

        if rank == 3:
            row, stack_size, filters_size = shape

            transpose_dimensions = (2, 1, 0)
            kernel_shape = (row,)
            correct_ifft = lambda shape, s=[None]: np.fft.irfft(shape, s[0])
            correct_fft = np.fft.rfft

        elif rank == 4:
            row, column, stack_size, filters_size = shape

            transpose_dimensions = (2, 3, 0, 1)
            kernel_shape = (row, column)
            correct_ifft = np.fft.irfft2
            correct_fft = np.fft.rfft2

        elif rank == 5:
            x, y, z, stack_size, filters_size = shape

            transpose_dimensions = (3, 4, 0, 1, 2)
            kernel_shape = (x, y, z)
            correct_fft = np.fft.rfftn
            correct_ifft = np.fft.irfftn
        else:
            return K.variable(self.orthogonal(shape), dtype=K.floatx())

        kernel_fourier_shape = correct_fft(np.zeros(kernel_shape)).shape

        init = []
        for i in range(filters_size):
            basis = self._create_basis(
                stack_size, np.prod(kernel_fourier_shape))
            basis = basis.reshape((stack_size,) + kernel_fourier_shape)

            filters = [correct_ifft(x, kernel_shape) +
                       np.random.normal(0, self.eps_std, kernel_shape) for
                       x in basis]

            init.append(filters)

        # Format of array is now: filters, stack, row, column
        init = np.array(init)
        init = self._scale_filters(init, variance)
        return init.transpose(transpose_dimensions)

    def _fit(self, _X, _y):


    def get_config(self):
        return {
            'seed': self.seed
        }
