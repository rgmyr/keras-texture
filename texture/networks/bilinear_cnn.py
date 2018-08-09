"""Bilinear Pooling CNN (BCNN) network as described in:

@inproceedings{lin2015bilinear,
    Author = {Tsung-Yu Lin, Aruni RoyChowdhury, and Subhransu Maji},
    Title = {Bilinear CNNs for Fine-grained Visual Recognition},
    Booktitle = {International Conference on Computer Vision (ICCV)},
    Year = {2015}
}
"""
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda, Flatten, Dense
from tensorflow.keras.models import Model as KerasModel

from texture.networks.util import make_backbone, make_dense_layers
from texture.ops import bilinear_pooling



def bilinear_cnn(fA, fB,
                 num_classes,
                 input_shape,
                 conv1x1=None,
                 dense_layers=[],
                 dropout_rate=None):
    '''Combine two feature extracting CNNs into single Model with bilinear_pooling + FC layers.
       fA and fB should output 4D tensors of equal shape, except (optionally) in # of channels.

    Parameters
    ----------
    fA : KerasModel or str
        Feature network A. Should output features (N, H, W, cA).
        If str, loads the corresponding ImageNet model from `keras.applications`.
    fB : KerasModel or str
        Feature network B. Should output features (N, H, W, cB).
        If str, loads the corresponding ImageNet model from `keras.applications`.
        If `None`, will return symmetric BCNN using fA.
    num_classes : int
            Number of classes for softmax output layer
    input_shape : tuple of int
        Shape of input images. Must be compatible with fA.input & fB.input.
    conv1x1 : int or iterable(int), optional
        Add a 1x1 conv to reduce number of channels in (fA, fB) to some value(s).
        If iterable, must be length 2; values then mapped to (fA, fB).
    dense_layers : iterable of int, optional
        Sizes for additional Dense layers between bilinear vector and softmax. Default=[].
    dropout_rate: float, optional
        Specify a dropout rate for Dense layers

    Returns
    -------
    B-CNN : KerasModel
        Single bilinear CNN composed from fA & fB (asymmetric) or fA with itself (symmetric)
    '''
    fA = make_backbone(fA)
    fB = make_backbone(fB)

    input_image = layers.Input(shape=input_shape)

    outA = fA(input_image)
    if fB is None:
        outB = outA             # symmetric B-CNN
    else:
        outB = fB(input_image)  # asymmetric B-CNN

    if isinstance(conv1x1, int):
        outA = Conv2D(conv1x1, (1,1), name='reduce_A')(outA)
        outB = Conv2D(conv1x1, (1,1), name='reduce_B')(outB)
    elif hasattr(conv1x1, '__iter__'):
        assert len(conv1x1) == 2, 'if iterable, conv1x1 must have length of 2'
        outA = Conv2D(conv1x1[0], (1,1), name='reduce_A')(outA)
        outB = Conv2D(conv1x1[1], (1,1), name='reduce_B')(outB)

    x = Lambda(bilinear_pooling, name='bilinear_pooling')([outA, outB])

    x = make_dense_layers(dense_layers, dropout_rate)(x)

    pred = Dense(num_classes, activation='softmax')(x)

    model = KerasModel(inputs=input_image, outputs=pred)

    return model
