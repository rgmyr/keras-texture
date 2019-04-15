"""Pre-processing and augmentation functions."""
import numpy as np
from skimage import transform, color
import matplotlib.pyplot as plt

import random
from math import floor


###+++++++++++++###
### IMAGE / VIZ ###
###+++++++++++++###

def center_crop(img, side_length):
    """
    Resize short side to side_length, then square center crop.
    """
    if img.ndim == 2:
       img = color.gray2rgb(img)
    h, w, _ = img.shape
    new_h, new_w = side_length, side_length
    if h > w:
        new_h = int(side_length*(h/w))
    else:
        new_w = int(side_length*(w/h))
    r_img = transform.resize(img, (new_h, new_w), mode='constant')

    h_offset = (new_h - side_length) // 2
    w_offset = (new_w - side_length) // 2

    return r_img[h_offset:h_offset+side_length,w_offset:w_offset+side_length]

def show_sample(dataset, split='train'):
    """
    Create figure with a random sample of 9 images from dataset.
    """
    if not hasattr(dataset, 'X_train'):
        dataset.load_or_generate_data()
    if split == 'train':
        X, y =  dataset.X_train, dataset.y_train
    else:
        X, y = dataset.X_test, dataset.y_test

    idxs = random.sample(np.arange(0, y.shape[0]).tolist(), 9)
    X_samples, y_samples = X[idxs], y[idxs]

    fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(20,20))
    for i, (xi, yi) in enumerate(zip(X_samples, y_samples)):
        r, c = floor(i/3), i % 3
        ax[r,c].set_xticks([])
        ax[r,c].set_yticks([])
        #if 'Facies' in dataset.__repr__():
        #    title = dataset.classes[np.argmax(yi)+2]
        #else:
        title = dataset.classes[np.argmax(yi)]
        ax[r,c].set_title(title)
        ax[r,c].imshow(xi)

    return fig


###++++++++++++++###
### DATA LOADING ###
###++++++++++++++###

def common_file_prefix(path_list):
    """
    Get common prefix substring of filenames in path_list.
    """
    char_zip = zip(*[p.split('/')[-1] for p in path_list])
    chars = []
    for tup in char_zip:
        if len(set(tup)) == 1:
            chars.append(tup[0])
        else:
            break
    return ''.join(chars)


def filter_child_dirs(root_path, conditional='training'):
    """
    Return a list of paths to all subdirs satisfying `conditional`.
    If `conditional` is a string, then matches on subdirs w/ dirname == `conditional`.
    """
    if isinstance(conditional, str):
        conditional = lambda p: p.split('/')[-1] == conditional

    return filter(conditional, [d[0] for d in os.walk(root_path)])
