#import h5py
import os
from glob import glob
import numpy as np
from skimage import io
from itertools import chain

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical

from texture.datasets.base import Dataset
from texture.datasets.util import center_crop


class FMDDataset(Dataset):
    """The Describable Textures Dataset contains 5640 images (47 classes with 120 examples each.)
    Authors release 10 different train/val/test splits (equally sized) for benchmarking.
    I'm just using train+val for X/y_train and test for X/y_test for any given split.

    Download links and more details at: https://www.robots.ox.ac.uk/~vgg/data/dtd/

    Parameters
    ----------
    data_dir : str
        Path to parent directory of unzipped dtd-r1.0.1.tar.gz
    input_size : int, optional
        Side length of squared input images (using datasets.util.center_crop), default=224
    random_seed : int, optional
        Random seed for train/test split, defualt=None
    """
    def __init__(self, data_dir, input_size=224, random_seed=None):
        self.num_classes = 10
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, 'image')
        self.seed = random_seed

        class_dirs = sorted(glob(os.path.join(self.img_dir, '*')))
        self.classes = sorted([s.split('/')[-1] for s in class_dirs])   # folders in image_dir should be class names
        print(self.classes)
        assert len(self.classes) == self.num_classes, "FMD data_dir should have 10 subdirs (== num_classes)"
        self.img_files = list(chain.from_iterable(glob(os.path.join(dir, '*.*g')) for dir in class_dirs))
        self.img_labels = np.array([self._to_class(fpath) for fpath in self.img_files])

        self.input_size = input_size
        self.input_shape = (input_size, input_size, 3)


    def data_dirname(self):
        return self.data_dir


    def load_or_generate_data(self):
        """Define X/y train/test."""
        X = np.array([center_crop(io.imread(fpath), self.input_size) for fpath in self.img_files])

        self.X_train, self.X_test, y_train, y_test = train_test_split(X, self.img_labels,
                                                                      test_size=0.1,
                                                                      stratify=self.img_labels,
                                                                      random_state=self.seed)

        self.y_train, self.y_test = to_categorical(y_train), to_categorical(y_test)


    def __repr__(self):
        return (
            'FMD Dataset\n'
            f'Num classes: {self.num_classes}\n'
            f'Random seed: {self.seed}\n'
            f'Input shape: {self.input_shape}\n'
        )


    def _to_class(self, s):
        return self.classes.index(s.split('/')[-2])
