import numpy as np
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# import dataset classes here
import texture.models as models_module
from texture.models import PredictorModel
#from corebreakout.datasets.generator import DatasetGenerator

DEFAULT_XBG_ARGS = {
    'max_depth' : 3,
    'learning_rate' : 0.1,
    'n_estimators' : 1000,
    'objective' : 'multi:softprob',
    'n_jobs' : 2,
    'gamma' : 0,
    'subsample' : 1,
    'colsample_bytree' : 1,
    'colsample_bylevel' : 1,
    'reg_alpha' : 0,                # L1 penalty
    'reg_lambda' : 1,               # L2 penalty
    'tree_method' : 'gpu_exact'
}

DEFAULT_SVC_ARGS = {
    'probability' : True
}

"""
Example of model_args[`feature_models`]:
{
    "image" : {
        "model" : "NetworkModel",
        "network" : "deepten",
        "optimizer" :
        "optimizer_args" :
    }
    "pseudoGR" : {
        "model" : "NetworkModel",
        "network" : "log_conv",
        "network_args" :
            {
            "n_layers"
            }
    }
    "logs" : {
        "model" : "ContextModel",
        "context_size" : 0.5
    }
}
Each key should be present in dataset_args['features'].

"""


class FeaturePredictor(PredictorModel):
    """Class for top level classifiers (operating on arbitrary 1D feature vectors).

    Parameters
    ----------
.
    model_args : dict, optional
        Parameters for constuctor & fit methods of chosen predictor type.
    """
    def __init__(self, dataset_cls, dataset_args={}, model_args={}, feature_model_args={}):

        PredictorModel.__init__(self, dataset_cls, dataset_args, model_args)

        model_type = self.model_args.pop('model_type', 'XGB')
        if 'XGB' in model_type:
            self.model_type = 'XGB'
            self.model_args = {**DEFAULT_XBG_ARGS, **model_args}
            self.model = XGBClassifier(**self.model_args)
        elif 'SVC' in model_type:
            self.model_type = 'SVC'
            self.model_args = {**DEFAULT_SVC_ARGS, **model_args}
            self.model = LinearSVC(**self.model_args)
        else:
            raise ValueError('`model_name` must contain one of {`XGB`, `SVC`}')

        feature_model_cls = getattr(models_module, feature_model_args['model'])
        self.feature_model = feature_model_cls(self.data, {}, feature_model_args['model_args'])
        self.feature_model_fit_args = feature_model_args.pop('fit_args', {})


    def fit(self, dataset, **fit_args):
        '''Must implement the fit method using only feature_models, dataset, & model_args.'''
        self.classes = dataset.classes
        self.feature_model.fit(dataset, **self.feature_model_fit_args)

        X_train = self.feature_model.extract_features(dataset.X_train)
        y_train = dataset.y_train.argmax(-1) if dataset.y_train.ndim > 1 else dataset.y_train

        self.model.fit(X_train, y_train, **fit_args)

        return self.evaluate(dataset.X_test, dataset.y_test, print_report=fit_args.pop('verbose', True))


    def predict(self, X):
        '''Must implement a predict method for input data X.'''
        X_features = self.feature_model.extract_features(X)
        return self.model.predict(X_features)


    def predict_proba(self, X):
        '''Class-wise probability predictions.'''
        X_features = self.feature_model.extract_features(X)
        return self.model.predict_proba(X_features)


    def evaluate(self, X, y, print_report=False):
        '''Return mean accuracy of predict(X).'''
        X_features = self.feature_model.extract_features(X)
        y_pred = self.model.predict(X_features)
        y_true = y.argmax(-1) if y.ndim > 1 else y
        acc = accuracy_score(y_true, y_pred)
        if print_report:
            print(classification_report(y_true, y_pred, target_names=self.classes))
            print("Total accuracy Score : ", acc)
        return acc


    def feature_importances(self):
        pass


    def collect_features(self, X):
        '''Get concatenated feature vectors for input data X.'''
        X_features = []
        for feature in self.features:
            X_features.append(self.feature_models[feature].extract_features(X))

        if not hasattr(self, 'features_info'):
            feature_sizes = [feats.shape[-1] for feats in X_features]
            self.features_info = {n : d for n, d in zip(self.features, feature_sizes)}

        return np.concatenate(X_features, axis=-1).squeeze()
