import numpy as np
from tensorflow.keras.optimizers import RMSprop

from texture.datasets.sequence import DatasetSequence


class Model:
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
    def __init__(self, dataset_cls, network_fn, dataset_args={}, network_args={}):
        self.name = f'{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}'

        self.data = dataset_cls{**dataset_args}

        self.network = network_fn(self.data.input_shape, self.data.output_shape, **network_args)
        self.network.summary()

        self.batch_augment_fn = None
        self.batch_format_fn = None

    @property
    def weights_filename(self, model_dir):
        return str(model_dir / f'{self.name}_weights.h5')

    def fit(self, dataset, batch_size=32, epochs=10, callbacks=[]):
        self.network.compile(loss=self.loss(), optimizer=self.optimizer(), metrics=self.metrics())

        train_seq = DatasetSequence(dataset.X_train, dataset.y_train, batch_size, augment_fn=self.batch_augment_fn, format_fn=self.batch_format_fn)
        test_seq = DatasetSequence(dataset.X_test, dataset.y_test, batch_size, augment_fn=self.batch_augment_fn, format_fn=self.batch_format_fn)

        self.network.fit_generator(
            train_seq,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=test_seq,
            use_multiprocesing=True,
            workers=1,
            shuffle=False # implemented Sequence.on_epoch_end() instead
        )

    def evaluate(self, X, y):
        seq = DatasetSequence(X, y. batch_size=16)
        preds = self.network.predict_generator(sequence)
        return np.mean(np.argmax(preds, -1) == np.argmax(y, -1))

    def loss(self):
        return 'categorical_crossentropy'

    def optimizer(self):
        return RMSprop()

    def metrics(self):
        return ['accuracy']

    def load_weights(self, model_dir):
        self.network.load_weights(self.weights_filename(model_dir))

    def save_weights(self, model_dir):
        self.network.save_weights(self.weights_filename(model_dir))
