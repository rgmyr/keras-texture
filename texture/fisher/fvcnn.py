import numpy as np
from cyvlfeat import gmm, fisher

from keras import applications
from keras.models import Model
from keras.layers import Lambda

from sklearn.svm import SVC, LinearSVC


__all__ = ['FVCNN']

# callables for pretrained model fetching
k_apps = {'vgg16'   : applications.vgg16.VGG16,
          'vgg19'   : applications.vgg19.VGG19,
          'resnet50': applications.resnet50.ResNet50}


class FVCNN():
    '''FV-CNN class fits a Gaussian Mixture Model to the output of a pretrained
       CNN on a new training set X, and uses the GMM to parameterize the Fisher
       Vector encoding of arbitrary inputs. Encodings may be generated directly,
       or to train a SVM classifier within the class instance.

    Parameters
    ----------
    CNN : `keras.models.Model` or str
        Either a `keras` Model, or one of {'vgg16','vgg19','resnet50',...}
        If str, loads the corresponding ImageNet model from `keras.applications`.
    k : int, optional
        Number of clusters to use for the GMM, default=32.
    X : np.array, shape (N,H,W,C), or list of np.array w/ shapes (H_i,W_i,C), optional
        A set of training images from which to generate the sample of CNN
        feature vectors to which a GMM will be fit. `C` should match the 
        `input_shape` of the CNN (must be 3 for ImageNet pretrained models).
        If X is given, the GMM will be fit automatically, otherwise it will
        must be fit by calling the fitGMM(X) method.
    '''
    def __init__(self, CNN, k=32, X=None):
        if isinstance(CNN, Model):
            assert len(CNN.output_shape)==4, 'CNN must output a 4D Tensor'
            self.cnn = CNN
        elif isinstance(CNN, str):
            assert CNN in k_apps.keys(), 'Invalid keras.applications string'
            self.cnn = k_apps[CNN](include_top=False)
        else:
            raise ValueError('CNN parameter for FVCNN has invalid type')
        
        self.k = k
        self.D = self.cnn.output_shape[-1]
        
        if X is not None:
            self.fit(X, self.k)

        return self

    def fit(self, X):
        '''Fit a GMM with `k` clusters using the sample images X.'''
        
        feats = self.cnn.predict(X)
        feats = np.reshape(feats, (-1, feats.shape[-1]))

        print('Fitting GMM with %d clusters...' % self.k)                # explore more params?
        self.means, self.covars, self.priors, self.LL, self.posteriors = gmm.gmm(feats, n_clusters=self.k)

        return self

    def encode(self, img):
        '''Generate the Fisher vector representation of an image.

        Parameters
        ----------
        img : np.array, shape (H,W,C)
            An input image

        Returns
        -------
        encoding : np.array, shape (2*k*D,)
            Encoding vector, where k=num_clusters and D is CNN output depth.
        '''
        assert hasattr(self, 'means'), 'must call `fit` before generating encodings'
        assert img.ndim() == 3, '`img` to be encoded must have 3 dimensions'

        feats = self.cnn.predict(img[np.newaxis,...])
        feats = np.reshape(feats, (-1, feats.shape[-1]))

        return fisher.fisher(feats.T, self.means.T, self.covars.T, self.priors.T)


    def train(self, X, y, svm='linear'):
        '''Train an SVM using a training set X, y.

        Parameters
        ----------
        X :
        y :
        svm : str, optional
            Specifies the kernel type to be used in the support vector classifier algorithm.  
            It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable. 
            If none or 'linear' is given, sklearn.svm.LinearSVC (liblinear) will be used, 
            otherwise sklearn.svm.SVC (libsvm) will be used.
            The former has more flexible penalty/loss options, and scales better to large numbers of
            samples (> ~10,000). The latter obviously has more flexibility in kernel types.

        Returns
        -------
        self.svc : LinearSVC or SVC object
        '''
                

        pass

    def predict(self, X):
        '''Use trained FVCNN.svc to predict classes for a batch of samples X.
        
        '''
        assert hasattr(self, 'svc'), 'Cannot predict before training classifier.'

        pass

