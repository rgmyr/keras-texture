#import h5py
import csv
import numpy as np
from skimage import io

from tensorflow.keras.utils import to_categorical

from texture.datasets.base import Dataset
from texture.datasets.util import center_crop


class DTDDataset(Dataset):
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
    split : int or str, optional
        Which split to use for train/test, default=1. Must be in [1,..,10].
    """
    def __init__(self, data_dir, input_size=224, split=1):
        self.data_dir = data_dir
        self.img_dir = data_dir + '/images/'

        self.input_size = input_size
        self.input_shape = (input_size, input_size, 3)

        self.num_classes = 47
        self.output_shape = (self.num_classes,)

        self.split = split
        train_reader = csv.reader(open(data_dir+'/labels/train'+str(split)+'.txt'))
        val_reader = csv.reader(open(data_dir+'/labels/val'+str(split)+'.txt'))
        test_reader = csv.reader(open(data_dir+'/labels/test'+str(split)+'.txt'))

        self.train_list = [line[0] for line in list(train_reader) + list(val_reader)]
        self.test_list = [line[0] for line in test_reader]

        self.classes = sorted(list(set([s.split('/')[0] for s in self.test_list])))

    def data_dirname(self):
        return self.data_dir

    def load_or_generate_data(self):
        """Define X/y train/test."""
        self.X_train = np.array([center_crop(io.imread(self.img_dir+f), self.input_size) for f in self.train_list])
        self.y_train = to_categorical(np.array([self._to_class(f) for f in self.train_list]), self.num_classes)

        self.X_test = np.array([center_crop(io.imread(self.img_dir+f), self.input_size) for f in self.test_list])
        self.y_test = to_categorical(np.array([self._to_class(f) for f in self.test_list]), self.num_classes)


    def __repr__(self):
        return (
            'DTD Dataset\n'
            f'Num classes: {self.num_classes}\n'
            f'Split: {self.split}\n'
            f'Input shape: {self.input_shape}\n'
        )

    def _to_class(self, s):
        return self.classes.index(s.split('/')[0])
