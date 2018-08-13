import numpy as np

from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import LearningRateScheduler

from texture.datasets.sequence import DatasetSequence
from imgaug import augmenters as iaa


aug_seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Affine(scale=(0.75, 1.25),
               translate_percent=(-1., 1.),
               mode='wrap'),
    #iaa.Sharpen(alpha=(0.0,0.25), lightness=(0.9, 1.1)),
    #iaa.Invert(0.5)
    iaa.CoarseDropout((0.1, 0.25), size_percent=(0.01, 0.1))
])

def train_batch_aug(batch_X, batch_y):
    return aug_seq.augment_images(batch_X), batch_y


class TextureModel:
    """Base class, to be subclassed by other predictor/dataset-specific types.

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
    """
    def __init__(self, dataset_cls, network_fn, dataset_args={}, network_args={}, optimizer_args={}):
        self.name = f'{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}'

        self.data = dataset_cls(**dataset_args)

        self.network = network_fn(self.data.num_classes, self.data.input_shape, **network_args)
        self.network.summary()
        self.optimizer_name = optimizer_args.pop('optimizer', 'Adam')
        self.optimizer_args = optimizer_args

        self.batch_augment_fn = train_batch_aug
        self.batch_format_fn = None

    @property
    def weights_filename(self, model_dir):
        return os.path.join(model_dir, 'saved_weights.h5')

    def fit(self, dataset, batch_size=32, epochs=1, callbacks=[]):
        self.network.compile(loss=self.loss(), optimizer=self.optimizer(callbacks), metrics=self.metrics())

        train_seq = DatasetSequence(dataset.X_train, dataset.y_train, batch_size, augment_fn=self.batch_augment_fn, format_fn=self.batch_format_fn)
        test_seq = DatasetSequence(dataset.X_test, dataset.y_test, batch_size, augment_fn=None, format_fn=self.batch_format_fn)

        self.network.fit_generator(
            train_seq,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=test_seq,
            use_multiprocessing=True,
            workers=1,
            shuffle=False # implemented Sequence.on_epoch_end() to do instance shuffle instead of just batch shuffle
        )

    def evaluate(self, X, y):
        eval_seq = DatasetSequence(X, y, batch_size=16)
        preds = self.network.predict_generator(eval_seq)
        return np.mean(np.argmax(preds, -1) == np.argmax(y, -1))

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

    def load_weights(self, model_dir):
        self.network.load_weights(self.weights_filename(model_dir))

    def save_weights(self, model_dir):
        self.network.save_weights(self.weights_filename(model_dir))
