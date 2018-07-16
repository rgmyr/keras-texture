import cyvlfeat
import numpy as np

from keras import applications
from keras.models import Model
from keras.layers import Lambda

# callables for pretrained model fetching
k_apps = {'vgg16'   : applications.vgg16.VGG16,
          'vgg19'   : applications.vgg19.VGG19,
          'resnet50': applications.resnet50.ResNet50}


class FVCNN():
    '''FV-CNN class fits a Gaussian Mixture Model to the output of a pretrained
       CNN on a new training set X, and uses the GMM to parameterize the Fisher
       Vector encoding of arbitrary inputs. Encodings may be generated directly,
       or to train a SVM within the class instance.

    Parameters
    ----------
    CNN : `keras.models.Model` or str
        Either a `keras` CNN Model, or one of {'vgg16','vgg19','resnet50',...}
        If the latter, loads corresponding model from `keras.applications`.
    X : np.array, shape (N,H,W,C)
        A set of training images from which to generate the sample of CNN
        feature vectors to which a GMM will be fit. `C` should be allowed
        by the CNN (must be 3 for ImageNet pretrained models)
    k : int, optional
        Number of clusters to use for the GMM, default=32.
    '''
    def __init__(self, CNN, X=None, k=32):
        if isinstance(CNN, Model):
            assert len(CNN.output_shape)==4, 'CNN must output a 4D Tensor'
            self.cnn = CNN
        elif isinstance(CNN, str):
            assert CNN in k_apps.keys(), 'Invalid keras.applications string'
            self.cnn = k_apps[CNN](include_top=False)
        else:
            raise ValueError('CNN parameter for FVCNN has invalid type')
        
        self.k = k
        
        if X is not None:
            self._fitGMM(X, self.k)

        return self

    def _fitGMM(self, X, k):
        '''Fit a GMM with `k` clusters to the samples X.'''
        features = self.cnn.predict(X)
        #self.mean, self.cov, self.priors = cyvlfeat.gmm(...)

        pass

    def encode(self, X):
        '''Encode a batch of images to a batch of Fisher vectors.

        Parameters
        ----------
        X : np.array, shape (N,H,W,D)

        Returns
        -------
        X_encoded : np.array, shape (N,2*k*D)
        '''
        pass
