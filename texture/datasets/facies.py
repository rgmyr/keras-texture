#import h5py
import csv
import numpy as np
from skimage import io

from tensorflow.keras.utils import to_categorical

from texture.datasets.base import Dataset
from texture.datasets.util import center_crop

lithology_classes = {
    'nc': 'no core',
    'bs': 'bad sand',
    's' : 'sand',
    'is': 'interbedded sand',
    'ih': 'interbedded shale',
    'sh': 'shale'
}


class FaciesPatchDataset(Dataset):
    """Class for individual patch classification of facies in processed core image datasets.

    Parameters
    ----------
    data_dir : str
        Path to parent directory of unzipped dtd-r1.0.1.tar.gz
    input_shape : tuple of int, optional
        Size of input patches for patch classification
    random_seed : int, optional
        Seed to use for train/test split, default=None.
    """
    def __init__(self, data_dir, input_shape=(100,800,3), split=1):
        self.data_dir = data_dir
        self.img_dir = data_dir + '/images/'

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
