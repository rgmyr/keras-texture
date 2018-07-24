##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Logistic Regression initializer for Dense (fully_connected) layers
##
## After initialization on a train set, may be used as kernel_initializer & bias_initailizer
##  
## This source file is licensed under the MIT-style license found in the
## LICENSE file in the root directory of the source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import absolute_import
import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.initializers import *
from keras.initializers import _compute_fans
from sklearn import linear_model


class LogReg(Initializer):
    """Initializer that generates weights for Dense layers by fitting a
    logistic regression model to a sample of (input, output) pairs.
    One expected use is for fitting a fully connected output layer
    to a bilinear pooling of pre-trained CNN features. Pooling output 
    is typically very large (e.g., 512*512), so that good initialization 
    can make convergence faster and easier to achieve.

    Data is expected to be large, so 'saga' solver is used. 

    Parameters
    ----------
    _X : array, shape (n_samples, n_features)
        Sample of input to the layer
    _y : array, shape (n_samples, n_classes)
        Corresponding sample of layer output (e.g., one-hot encoded `y`)
    input_model : tf.keras.models.Model instance or `None`
        If an input model is provided, it is assumed that the sample inputs 
        to the layer will be input_model.predict(_X). Otherwise, it is assumed
        that _X should be interpreted as inputs to the layer. Default=`None`.
    penalty : str, 'l1' or 'l2', default='l2'
        Norm to use in the regularization penalty
    seed : int, default=None
        Used to seed the random generator.
    n_jobs : int, default=-1
        Number of CPU cores for parallelizing over classes. Value of -1 uses all cores.
    """

    def __init__(self, _X, _y, input_model=None, penalty='l2', seed=None, n_jobs=-1):
        if _X is None or _y is None:
            raise ValueError('Must pass _X and _y to construct LogReg initializer')
        if input_model is not None:
            _X = input_model.predict(_X)
        self.penalty = penalty
        self.seed = seed
        self.n_jobs = n_jobs
        self._fit(_X, _y)


    def __call__(self, shape, dtype=None):
        """Return model coef_ or bias, provided `shape` matches."""

        rank = len(shape)
        assert rank in [1,2], 'LogReg can only be used to initialize Dense-like kernels & biases' 

        n_class, n_feat = self.model.coef_.shape

        if rank == 1:
            assert shape == (n_class,), '1D `shape` should match LogReg.model.intercept_'
            return tf.convert_to_tensor(self.model.intercept_.astype(np.float32), dtype=dtype)

        elif rank == 2:
            assert shape == (n_feat, n_class), '2D `shape` should match LogReg.model.coef_.T'
            return tf.convert_to_tensor(self.model.coef_.T.astype(np.float32), dtype=dtype)


    def _fit(self, _X, _y):
        """Fit the logistic regression model."""

        self.model = linear_model.LogisticRegression(penalty=self.penalty, random_state=self.seed,
                                                    solver='saga', n_jobs=self.n_jobs)
        self.model.fit(_X, _y)


    def get_config(self):
        return {
            'penalty': self.penalty,
            'seed': self.seed,
            'n_jobs': self.n_jobs
        }
