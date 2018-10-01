import numpy as np

from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import save_model as save_KerasModel

import texture.networks as networks_module
from texture.datasets.sequence import DatasetSequence
from texture.models import FeatureModel, PredictorModel

from imgaug import augmenters as iaa


DEFAULT_FIT_ARGS = {
    'epochs' : 1,
    'batch_size' : 1,
    'callbacks' : []
}

aug_seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Affine(scale=(0.75, 1.25),
               translate_percent=(-1., 1.),
               mode='wrap'),
    #iaa.Sharpen(alpha=(0.0,0.25), lightness=(0.9, 1.1)),
    #iaa.Invert(0.5)
    #iaa.CoarseDropout((0.0, 0.25), size_percent=(0.02, 0.2))
])

def train_batch_aug(batch_X, batch_y):
    return aug_seq.augment_images(batch_X), batch_y


class NetworkModel(FeatureModel, PredictorModel):
    """Network class.

    Parameters
    ----------
    dataset_cls : type
        type of Dataset class to model on
    network_fn : Callable
        Function returning a KerasModel instance
    dataset_args : dict, optional
        Keyword args for dataset_cls constructor
    network_args : dict, optional
        Keyword args for network_fn
    optimizer_args : dict, optional
        Keyword args to specify optimizer + associated params
    """
    def __init__(self, dataset_cls, dataset_args={}, model_args={}):
        PredictorModel.__init__(self, dataset_cls, dataset_args, model_args)

        self.name = f'{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}'

        network_fn = getattr(networks_module, self.model_args['network'])
        network_args = self.model_args.get('network_args', {})
        self.network = network_fn(self.data.num_classes, self.data.input_shape, **network_args)
        self.network.summary()

        self.optimizer_args = self.model_args.get('optimizer_args', {})
        self.optimizer_name = self.optimizer_args.pop('optimizer', 'Adam')
        self.fit_args = {**DEFAULT_FIT_ARGS, **self.model_args.get('fit_args', {})}

        self.batch_augment_fn = train_batch_aug
        self.batch_format_fn = None

    @property
    def model_filename(self, model_dir):
        return os.path.join(model_dir, 'saved_model.h5')

    def fit(self, dataset, **fit_args): # batch_size=32, epochs=1, callbacks=[]):

        callbacks = fit_args.get('callbacks', [])
        self.network.compile(loss=self.loss(), optimizer=self.optimizer(callbacks), metrics=self.metrics())

        train_seq = DatasetSequence(dataset.X_train, dataset.y_train, self.fit_args['batch_size'],
                                    augment_fn=self.batch_augment_fn, format_fn=self.batch_format_fn)
        test_seq = DatasetSequence(dataset.X_test, dataset.y_test, self.batch_size['batch_size'],
                                   augment_fn=None, format_fn=self.batch_format_fn)

        hist = self.network.fit_generator(
            train_seq,
            epochs=self.fit_args['epochs'],
            callbacks=callbacks,
            validation_data=test_seq,
            use_multiprocessing=True,
            workers=1,
            shuffle=False # implemented Sequence.on_epoch_end() to do instance shuffle instead of just batch shuffle
        )

        self.val_loss = hist.history['val_loss']


    def predict(self, X):
        return np.argmax(self.network.predict(X), -1)

    def predict_proba(self, X):
        return self.network.predict(X)

    def evaluate(self, X, y):
        eval_seq = DatasetSequence(X, y, batch_size=16)
        preds = self.network.predict_generator(eval_seq)
        return np.mean(np.argmax(preds, -1) == np.argmax(y, -1))


    def extract_features(self, X):
        '''Return features for inputs X (assumed that penultimate layer = features).'''
        try:
            return self.feature_network.predict(X, batch_size=1, verbose=1)
        except AttributeError:
            self.feature_network = KerasModel(inputs=self.network.input, outputs=self.network.layers[-2].output)
            return self.feature_network.predict(X, batch_size=1, verbose=1)


    def loss(self):
        return 'categorical_crossentropy'


    def optimizer(self, callbacks):
        # Using SGD if LRSchedule present, since it plays nicer than stateful opts
        if any([isinstance(callback, LearningRateScheduler) for callback in callbacks]):
            print("Found LearningRateScheduler callback --> using SGD Optimizer")
            return getattr(optimizers, 'SGD')(**self.optimizer_args)
        else:
            return getattr(optimizers, self.optimizer_name)(**self.optimizer_args)


    def metrics(self):
        return ['accuracy']

    def load_model(self, model_dir, compile=True):
        load_KerasModel(self.model_filename(model_dir), compile=compile)

    def save_model(self, model_dir):
        save_KerasModel(self.network, self.model_filename(model_dir))
