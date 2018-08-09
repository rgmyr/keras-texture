import numpy as np
import pandas as pd
from skimage import io

from tensorflow.keras.utils import to_categorical

from texture.datasets.base import Dataset
from texture.datasets.util import center_crop

class BirdsDataset(Dataset):

    def __init__(self, data_dir, input_size=448):
        
        self.data_dir = data_dir

        self.input_size = input_size
        self.input_shape = (input_size, input_size, 3)

        self.num_classes = 200
        self.output_shape = (self.num_classes,)

        img_dir = data_dir + '/images/'
        img_files = pd.read_csv(data_dir+'/images.txt', index_col=0, header=None, sep=' ')[1].values
        self.img_paths = img_dir + img_files
        self.img_labels = np.loadtxt(data_dir+'/image_class_labels.txt',dtype=np.uint8)[:,1]

        self.idxs = np.loadtxt(args.datapath+'/train_test_split.txt',dtype=np.uint8)[:,1]
        self.train_ix = np.where(idxs == 1)[0]
        self.test_ix  = np.where(idxs == 0)[0]


    def data_dirname(self):
        return self.data_dir


    def load_or_generate_data(self):

        X_train = np.array([center_crop(io.imread(self.img_paths[i]), input_size) for i in self.train_ix])
        y_train = [self.img_labels[i] for i in self.train_ix]

        X_test = np.array([center_crop(io.imread(self.img_paths[i]), input_size) for i in self.test_ix])
        y_test = [self.img_labels[i] for i in self.test_ix]

        y_train, y_test = to_categorical(y_train), to_categorical(y_test)

