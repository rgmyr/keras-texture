import numpy as np
from cyvlfeat import gmm, fisher

from keras import applications
from keras.models import Model
from keras.layers import Lambda

from sklearn.svm import SVC, LinearSVC

__all__ = ['FisherEncoder']


# callables for pretrained model fetching
k_apps = {'vgg16'               : applications.vgg16.VGG16,
          'vgg19'               : applications.vgg19.VGG19,
          'resnet50'            : applications.resnet50.ResNet50,
          'xception'            : applications.xception.Xception,
          'mobilenet'           : applications.mobilenet.MobileNet,
          'mobilenetv2'         : applications.mobilenetv2.MobileNetV2,
          'densenet121'         : applications.densenet.DenseNet121,
          'densenet169'         : applications.densenet.DenseNet169,
          'densenet201'         : applications.densenet.DenseNet201,
          'nasnet_large'        : applications.nasnet.NASNetLarge,
          'nasnet_mobile'       : applications.nasnet.NASNetMobile,
          'inception_v3'        : applications.inception_v3.InceptionV3,
          'inception_resnet_v2' : applications.inception_resnet_v2.InceptionResNetV2}


class FisherEncoder(FeatureModel):
    '''FV-CNN class fits a Gaussian Mixture Model to the output of a pretrained CNN
    on a new training set X, and uses the GMM to parameterize the Fisher vector encoding
    of arbitrary inputs. Encodings may be generated directly,or to train a SVM classifier
    within the class instance.

    Parameters
    ----------
    CNN : `keras.models.Model` or str
        Either a `keras` Model instance, or one of {'vgg16','vgg19','resnet50',...}
        If str, loads the corresponding ImageNet model from `keras.applications`.
    k : int, optional
        Number of clusters to use for the GMM, default=64.
    '''
    def __init__(self, CNN, k=64):
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


    def fit(self, X, y, gmm_init='kmeans', svc_kernel='linear', svc_penalty='l2', C=1.0, seed=None):
        '''Fit a GMM with `k` clusters using the sample images `X`. Then train a SVC on
        on the Fisher vector encodings of `X`, given the class labels `y`

        Parameters
        ----------
        X : array, shape (N,H,W,C), or list of N arrays w/ shapes (H_i,W_i,C)
            A set of training images from which to generate a sample of CNN
            feature vectors to which a GMM will be fit. `C` should match the
            `input_shape` of the CNN (must be 3 for ImageNet pretrained models).
        y : array, shape (N,)
            Array of class labels of images in `X`.
        gmm_init : str, optional
            Method to use for GMM initialization. One of {'kmeans', 'rand'}. Default = 'kmeans'.
            Custom init is also possible through `cyvlfeat`, but is not implemented here.
        svc_kernel : str, optional
            Specifies the kernel type to be used in the support vector classifier algorithm.
            It must be one of {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or a callable.
            If none or 'linear' is given, sklearn.svm.LinearSVC (liblinear) will be used,
            otherwise sklearn.svm.SVC (libsvm) will be used. (Actually, just LinearSVC for now.)
            The former has more flexible penalty/loss options, and scales better to large numbers of
            samples (> ~10,000). The latter obviously has more flexibility in kernel types.
        svc_penalty : str, optional
            One of {`l1`, `l2`}. Default='l2' is recommended.
        C : float, optional
            Inverse of regularization strength. Default=1.0 seems to work best in VGG+DTD tests.
        seed : int, optional
            Specify a random state for deterministic results. Default=None.

        Returns
        -------
        train_score : float
            Mean accuracy of trained SVC on the training set (`X`, `y`)
        '''
        if isinstance(X, np.ndarray):
            assert X.ndim==4, 'X must have shape (N,H,W,C) if an np.array'
            feats = self.cnn.predict(X)
            feats = feats.reshape(-1, feats[-1])
        elif isinstance(X, list):
            assert isinstance(X[0], np.ndarray), 'X must contain numpy.ndarrays, if a list'
            img_feats = [self._localfeatures(x) for x in X]
            #print('(sample of) img_feats.shapes:', [i.shape for i in img_feats[0:5]])
            feats = np.vstack(img_feats)
            #print('all_feats.shape :', feats.shape)
        else:
            raise ValueError('GMM input X has unknown form. Should be 4D array or list of 3D arrays.')

        # Fit the GMM
        # TODO: figure out covariance_bound Buffer bug
        #       should be = to max(all_feats.var(axis=k_feat))*0.0001
        print('Fitting GMM with %d clusters...' % self.k)
        self.means, self.covars, self.priors, LL, posteriors = gmm.gmm(feats, n_clusters=self.k,
                                                                       covariance_bound=None,
                                                                       init_mode=gmm_init)
        # Train the SVC
        if svc_kernel=='linear':
            self.svc = LinearSVC(penalty=svc_penalty, C=C, class_weight='balanced', random_state=seed)
        else:
            raise NotImplementedError('Only `linear` svc_kernel implemented right now.')

        fv_X = self.encode_batch(img_feats)

        self.svc.fit(fv_X, y)

        return self.svc.score(fv_X, y)


    def score(self, X_test, y_test):
        '''Get mean accuracy of SVC on test set.

        Parameters
        ----------
        X : array, shape (N,H,W,C), or list of N arrays w/ shapes (H_i,W_i,C)
            A set of images for which to predict classes from FV encodings.
        y : array, shape (N,)
            Ground truth class labels for all images in `X`

        Returns
        -------
        score : float
            Mean accuracy for predictions y_hat relative to y
        '''
        assert hasattr(self, 'svc'), 'Cannot predict before training the SVC.'

        fv_X = self.encode_batch(X_test)

        return self.svc.score(fv_X, y_test)


    def predict(self, X):
        '''Use trained FVCNN.svc to predict classes for a batch of samples X.

        Parameters
        ----------
        X : array, shape (N,H,W,C), or list of N arrays w/ shapes (H_i,W_i,C)
            A set of images for which to predict classes from FV encodings.

        Returns
        -------
        preds : array, shape (N,)
            Predicted class labels for all images in `X`
        '''
        assert hasattr(self, 'svc'), 'Cannot predict before training the SVC.'

        fv_X = self.encode_batch(X)

        return self.svc.predict(fv_X)

    def encode(self, x, verbose=False):
        '''Generate the Fisher vector representation of an image or its feature vectors.

        Parameters
        ----------
        x : array, shape (H,W,C) or (M,D)
            An input image, or a set of local feature vectors for the image.
            If (H,W,D), the (M,D) feature vectors are extracted using the CNN.
        verbose : bool, optional
            Whether to print FV information from VLFeat call, default=False.

        Returns
        -------
        encoding : array, shape (2*k*D,)
            FV encoding of img, where k=num_clusters in GMM and D is CNN output depth.
        '''
        assert hasattr(self, 'means'), 'Must call `fit` before generating encodings'

        if x.ndim == 2 and x.shape[-1] == self.D:
            feat = x
        elif x.ndim == 3:
            feat = self._localfeatures(x)
        else:
            raise ValueError('`img` to be encoded must be 3D image or 2D feature vectors')

        return fisher.fisher(feat.T, self.means.T, self.covars.T, self.priors.T,
                             improved=True, fast=False, verbose=verbose)


    def encode_batch(self, X):
        '''Encode 4D image batch array or list of 3D image arrays.'''
        if isinstance(X, np.ndarray):
            N = X.shape[0]
        else:
            N = len(X)

        fv_X = np.zeros((N, 2*self.k*self.D))
        for i in range(N):
            if isinstance(X, np.ndarray):
                x_i = X[i,...]
            else:
                x_i = X[i]
            fv_X[i,:] = self.encode(x_i)

        return fv_X

    def _localfeatures(self, x):
        '''Get reshaped CNN features for an image `x`.'''
        if x.ndim == 3:
            x = x[np.newaxis,...]
        feat = self.cnn.predict(x)
        return feat.reshape(-1,feat.shape[-1])
