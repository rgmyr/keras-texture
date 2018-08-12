import numpy as np
from tensorflow.keras.utils import Sequence


class DatasetSequence(Sequence):
    """
    Minimal subclassing of Sequence for use with fit_generator.
    """
    def __init__(self, X, y, batch_size=32, augment_fn=None, format_fn=None):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment_fn = augment_fn
        self.format_fn = format_fn

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        # idx = 0 -- uncomment to overfit a single batch
        begin = idx * self.batch_size
        end = (idx+1) * self.batch_size

        batch_X = self.X[begin:end]
        batch_y = self.y[begin:end]

        if batch_X.dtype  == np.uint8:
            batch_X = (batch_X / 255.).astype(np.float32)

        if self.augment_fn:
            batch_X, batch_y = self.augment_fn(batch_X, batch_y)

        if self.format_fn:
            batch_X, batch_y = self.format_fn(batch_X, batch_y)

        return batch_X, batch_y

    def on_epoch_end(self):
        """Shuffle the examples, not just batches via fit_generator."""
        p = np.random.permutation(len(self.X))
        self.X = self.X[p]
        self.y = self.y[p]
