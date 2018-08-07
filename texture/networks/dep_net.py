"""

"""
import tensorflow as tf
from tensorflow.keras.layers import Input, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model as KerasModel

from texture.layers import Encoding, BilinearModel
from texture.networks.util import make_backbone

def dep_net(backbone_cnn,
            num_classes,
            input_shape,
            encode_K=8,
            encode_feats=64,
            pooled_feats=64,
            dense_feats=128,
            dropout_rate=None):
    '''Combine a backbone CNN + Encoding layer + Dense layers into a DeepTEN.

    Parameters
    ----------
    backbone_cnn : KerasModel or str
        Feature extraction network. If KerasModel, should output features (N, H, W, C).
        If str, loads the corresponding ImageNet model from `keras.applications`.
    n_classes : int
        Number of classes for softmax output layer
    input_shape : tuple of int
        Shape of input image. Can be None, since Encoding layer allows variable input sizes.
    encode_K : int, optional
        Number of codewords to learn, default=8.
    encode_feats : int, optional
        Number of output nodes in post-Encoding/pre-BilinearModel Dense layer, default=64.
    pooled_feats : int, optional
        Number of output nodes in post-pooling/pre-BilinearModel Dense layers, default=64.
    dense_feats : int, optional
        Number of output nodes for penultimate Dense layer, default=128.
    dropout_rate: float, optional
        Specify a dropout rate for Dense layers

    Returns
    -------
    DEPnet : KerasModel
        Deep Encoding Pooling Network
    '''
    backbone_model = make_backbone(backbone_cnn)
    input_image = Input(shape=input_shape)
    conv_output = backbone_model(input_image)

    encode_head = BatchNormalization()(conv_output)
    encode_head = Encoding(encode_K, dropout=dropout_rate)(encode_head)
    encode_head = Dense(encode_feats, activation='relu')(encode_head)

    pooled_head = GlobalAveragePooling2D()(conv_output)
    pooled_head = BatchNormalization()(pooled_head)
    pooled_head = Dense(pooled_feats, activation='relu')(pooled_head)

    if dropout_rate is not None:
        encode_head = Dropout(rate=dropout_rate)(encode_head)
        pooled_head = Dropout(rate=dropout_rate)(pooled_head)

    output_head = BilinearModel()([encode_head, pooled_head])  # flat, l2_normalized output by default
    output_head = Dense(dense_feats, activation='relu')(output_head)
    output_head = tf.nn.l2_normalize(output_head, axis=-1)

    pred = Dense(num_classes, activation='softmax')(output_head)

    model = KerasModel(inputs=input_image, outputs=pred)

    return model
