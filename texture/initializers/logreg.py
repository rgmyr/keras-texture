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
    penalty : str, 'l1' or 'l2', default='l2'
        Norm to use in the regularization penalty
    seed : int, default=None
        Used to seed the random generator.
    """

    def __init__(self, _X, _y, seed=None):
        if _X is None or _y is None:
            raise ValueError('Must pass _X and _y to use LogReg initializer')
        self.eps_std = eps_std
        self.seed = seed
        self.orthogonal = Orthogonal()

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

    def _create_basis(self, filters, size):
        if size == 1:
            return np.random.normal(0.0, self.eps_std, (filters, size))

        nbb = filters // size + 1
        li = []
        for i in range(nbb):
            a = np.random.normal(0.0, 1.0, (size, size))
            a = self._symmetrize(a)
            u, _, v = np.linalg.svd(a)
            li.extend(u.T.tolist())
        p = np.array(li[:filters], dtype=K.floatx())
        return p

    def _symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())

    def _scale_filters(self, filters, variance):
        c_var = np.var(filters)
        p = np.sqrt(variance / c_var)
        return filters * p

    def get_config(self):
        return {
            'eps_std': self.eps_std,
            'seed': self.seed
        }
