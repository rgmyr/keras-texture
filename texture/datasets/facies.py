#import h5py
import numpy as np
import pandas as pd
from skimage import io

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from texture.datasets.base import Dataset
from texture.datasets.util import filter_child_dirs, common_file_prefix


lithology_classes = {
    'nc': 'no core',
    'bs': 'bad sand',
    's' : 'sand',
    'is': 'interbedded sand',
    'ih': 'interbedded shale',
    'sh': 'shale'
}


def make_patches(img, labels, patch_height, bad_labels=[0,1], stride=None, shuffle=False):
    '''Generate (patch, label) pairs.
    Parameters
    ----------
    img : np.array
        Core column image array
    labels : 1-D np.array
        Row-wise label array
    patch_height : int
        Desired patch height, in pixels
    stride : int, optional
        (Optional) Stride between patches (patch_height - stride = overlap). Default = patch_height
    shuffle : bool, optional
        (Optional) Whether to shuffle patche/label pairs before returning
    Returns
    -------
    X, y:
        Tuple of np.array with shapes: (num_patches, patch_height, patch_width, 3), (num_patches)
    TODO: support for mixed labels?
    '''
    step = patch_height if stride is None else stride
    img_h, img_w = img.shape[0], img.shape[1]

    n_patches = math.floor(img_h / step)
    X = np.zeros((n_patches, patch_height, img_w, 3), dtype=img.dtype)
    y = np.zeros(n_patches, dtype=np.uint8)
    good_idxs = []

    print("Making ", n_patches, " patches...")
    indices = list(range(n_patches))
    if shuffle:
        random.shuffle(indices)

    r0 = 0
    for i in indices:
        r1 = r0 + step
        X[i,...] = img[r0:r1,:]
        y[i] = mode(labels[r0:r1])[0][0]
        if i not in bad_labels:
            good_idxs.append(i)
        r0 += step

    X = X[good_idxs]
    y = y[good_idxs]
    y -= y.min()
    counts = np.bincount(y)
    print("Label counts = ", list(zip(range(1,len(counts)), counts[1:])))

    return X, y


class FaciesPatchDataset(Dataset):
    """Class for individual patch classification of facies in processed core image datasets.
    Each well should have its own directory, with an image file, a depths file, and a labels file.
    Each of the files should have the same prefix (name of the well) and suffixes corresponding
    to the `img_ext`, `depth_ext` and `labels_ext` arguments.


    Parameters
    ----------
    data_dir : str
        Common parent path of one or more `subdirs` directories containing modeling data.
    subdirs : str
        Name of subdirectories that contain the modeling data (from an individual well).
    input_shape : tuple of int, optional
        Size of input patches for patch classification
    split : one of {'by_well', 'by_patch'}, optional
        Whether to split train/test by well or by individual image patches, default='by_patch'.
    random_seed : int, optional
        Seed to use for train/test split, default=None.
    """
    def __init__(self,
                data_dir,
                subdirs,
                img_ext='img.png',
                depth_ext='depth.npy'
                labels_ext='labels.csv',
                lithology_classes=lithology_classes,
                input_shape=(100,800,3),
                split='by_patch',
                random_seed=None):
        # arg params
        self.data_dir = data_dir
        self.input_shape = input_shape
        self.split = split
        self.seed = random_seed

        # classes info
        self.labels = list(lithology_classes.keys())
        self.classes = list(lithology_classes.values())
        self.num_classes = len(self.classes)
        self.output_shape = (self.num_classes,)

        # load data from each good subdir
        self.data_dict = {}
        for d in filter_child_dirs(subdirs, data_dir):
            img_file = glob.glob(os.path.join(d, '*'+img_ext))[0]
            depth_file = glob.glob(os.path.join(d, '*'+depth_ext))[0]
            labels_file = glob.glob(os.path.join(d, '*'+labels_ext))[0]

            well_name = common_file_prefix([img_file, depth_file, labels_file])
            assert len(well_name) > 0, 'files in well need to share a common prefix'
            assert well_name not in data_dict.keys(), 'well_names must be unique'

            if img_file[-3:] == 'png':
                img = io.imread(img_file)
            elif img_file[-3:] == 'npy':
                img = np.load()
            depth = np.load(depth_file)
            self.data_dict[well_name] = {
                'image': img,
                'depth': depth,
                'labels': self.label_rows(depth, pd.read_csv(labels_file))
            }

    def data_dirname(self):
        return self.data_dir


    def load_or_generate_data(self):
        """Define X/y train/test."""
        # ASSUMING by_patch for now
        patch_sets = []
        label_sets = []
        for well_name, data in self.data_dict.items():
            img = self.crop_or_pad_image(data['image'], self.input_shape[1])
            X, y = make_patches(img, data['labels'], self.input_shape[0])  # add opts here
            patch_sets.append(X)
            label_sets.append(y)

        X = np.vstack(patch_sets)
        y = np.vstack(label_sets)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y,
                                                                                test_size=0.2,
                                                                                random_state=self.seed,
                                                                                stratify=y)
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)
        

    def label_rows(self, depth, df):
        '''Convert interval labels to image row labels.'''
        labels = np.zeros_like(depth, dtype=np.uint8)
        ix = 0
        row = df.iloc[ix]
        for i in range(depth.size):
            if depth[i] > row.base:
                ix += 1
                row = df.iloc[ix]
            labels[ix] = self.labels.index(row.lithology)
        return labels


    def crop_or_pad_image(self, img, width):
        '''Make image_width = width by center cropping or padding.'''
        width_diff = width - img.shape[1]
        if width_diff == 0:
            return img
        else:
            l = abs(width_diff) // 2
            r = abs(width_diff) // 2 + abs(width_diff) % 2
            # center crop if img too wide
            if width_diff < 0:
                return img[:,l:-r,:]
            # zero pad if img too narrow
            else:
                lhs = np.zeros((img.shape[0],l,3), dtype=img.dtype)
                rhs = np.zeros((img.shape[0],r,3), dtype=img.dtype)
                return np.hstack([lhs, img, rhs])

        elif width_diff < 0:



    def __repr__(self):
        return (
            'Facies Patch Dataset\n'
            f'Classes: {self.classes}\n'
            f'Split type: {self.split}\n'
            f'Input shape: {self.input_shape}\n'
        )


    def _to_class(self, s):
        return self.classes.index(s.split('/')[0])
