"""Pre-processing and augmentation functions."""
import numpy as np
from skimage import transform, color

def center_crop(img, side_length):
    '''Resize short side to side_length, then square center crop.'''
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
