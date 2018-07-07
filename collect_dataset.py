import argparse
import numpy as np
import pandas as pd
from skimage import io, transform, color
from keras import utils

valid_datasets = ['birds'] #,'cars','airplanes']

def l2s(slist):
    return '{'+','.join(slist)+'}'

parser = argparse.ArgumentParser(description='Train B-CNN built on VGG-M/D')
parser.add_argument('--dataset', dest='dataset', required=True,
                    help='Name of benchmark dataset. One of '+l2s(valid_datasets))
parser.add_argument('--datapath', dest='datapath', required=True,
                    help='Path to root folder of dataset')
args = parser.parse_args()

assert args.dataset in valid_datasets, 'dataset must be valid'


def crop_and_resize(img, resize_shape):
    if img.ndim == 2:
       img = color.gray2rgb(img) 
    h, w, _ = img.shape
    diff = np.abs(h-w)
    start = diff // 2
    if diff == 1:
        img = img
    elif h > w:
        img = img[start:-start,:,:]
    elif w > h:
        img = img[:,start:-start,:]
    return transform.resize(img, resize_shape)

if args.dataset == 'birds':
    n_classes = 200
    input_shape = (448, 448, 3)

    img_dir = args.datapath + 'images/'
    img_paths = img_dir + pd.read_csv(args.datapath+'images.txt', index_col=0, header=None, sep=' ')[1]
    img_paths = img_paths.values

    img_labels = np.loadtxt(args.datapath+'image_class_labels.txt',dtype=np.uint8)[:,1]

    idxs = np.loadtxt(args.datapath+'train_test_split.txt',dtype=np.uint8)[:,1]
    train_ix = np.where(idxs == 1)[0]
    test_ix  = np.where(idxs == 0)[0]

    X_train, y_train = np.empty((len(train_ix), 448, 448, 3)), np.zeros_like(train_ix)
    X_test , y_test  = np.empty((len(test_ix) , 448, 448, 3)), np.zeros_like(test_ix)

    for i, idx in enumerate(list(train_ix)):
        #print(img_paths[idx])
        img = io.imread(img_paths[idx])
        X_train[i,...] = crop_and_resize(img, input_shape)
        y_train[i] = img_labels[idx]
    for i, idx in enumerate(list(test_ix)):
        img = io.imread(img_paths[idx])
        X_test[i,...] = crop_and_resize(img, input_shape)
        y_test[i] = img_labels[idx]

    y_train = utils.to_categorical(y_train-1, n_classes)
    y_test = utils.to_categorical(y_test-1, n_classes)


    np.save('birds_X_train.npy', X_train)
    np.save('birds_y_train.npy', y_train)

    np.save('birds_X_test.npy', X_test)
    np.save('birds_y_test.npy', y_test)

