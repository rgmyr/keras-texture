from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract base class for all <>Model subclasses.

    Parameters
    ----------
    dataset_cls : Dataset type or instance
        If type, instantiated with dataset_args, otherwise treated as existing Dataset.
    dataset_args : dict
        Arguments for Dataset constructor, if necessary.
    model_args : dict
        Arguments for the Model subclass, saved as an attribute.
    """
    def __init__(self, dataset_cls, dataset_args, model_args):
        if 'from_file' in model_args.keys():
            self.load(model_args['from_file'])
            self.was_loaded_from_file = True

        else:
            self.was_loaded_from_file = False
            self.model_args = model_args
            if isinstance(dataset_cls, type):
                self.data = dataset_cls(**dataset_args)
            else:
                self.data = dataset_cls

    @abstractmethod
    def fit(self, dataset, **fit_args):
        pass

    @classmethod
    def save(self, path):
        path = str(path)
        if path.endswith('.pkl'):
            pkl_file = open(path, 'wb')
            pickle.dump(self.__dict__, pkl_file)
        elif '.' in path.split('/')[-1]:
            raise ValueError('model.save/load path must be a stem or a `.pkl` file')
        else:
            pkl_file = open(path + '.pkl', 'wb')
            pickle.dump(self.__dict__, pkl_file)
        pkl_file.close()

    @classmethod
    def load(self, path):
        path = str(path)
        if path.endswith('.pkl'):
            pkl_file = open(path, 'rb')
            self.__dict__.update(pickle.load(pkl_file))
        elif '.' in path.split('/')[-1]:
            raise ValueError('model.save/load path must be a stem or a `.pkl` file')
        else:
            pkl_file = open(path + '.pkl', 'rb')
            self.__dict__.update(pickle.load(pkl_file))
        pkl_file.close()


class FeatureModel(BaseModel):
    """
    Base class for Models that can provide features from raw data to Predictors.
    """
    @abstractmethod
    def extract_features(self, X):
        pass


class PredictorModel(BaseModel):
    """
    Base class for Models that can perform (probabalistic) class prediction.

    Parameters
    ----------
    dataset_cls : Dataset type, or instance thereof
        If type, instantiated with dataset_args, else treated as existing Dataset.
    dataset_args : dict, optional
        Arguments for Dataset constructor, if necessary

    """
    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    @abstractmethod
    def evaluate(self, X, y):
        pass
