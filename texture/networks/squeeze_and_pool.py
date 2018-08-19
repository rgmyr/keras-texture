
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Conv2D, BatchNormalization, Concatenate, MaxPool2D, Input, Lambda, Dense

#from texture.initializers import ConvolutionAware
from texture.networks.util import make_dense_layers
from texture.ops import bilinear_pooling


def squeeze_and_pool(x, r, pool_type='bilinear', dropout_rate=None):

    C = tf.shape(x)[-1]

    # squeeze
    z = GlobalAveragePooling2D()(x)
    z = Dense(int(C/r), activation='relu')(z)

    # save squeeze output for pooling
    x_pooling = pool_type()(z)

    #excite
    s = Dense(C, activation='sigmoid')(z)

    u = tf.multiply(x, s)

    return u, x_pooling


def SP_block(_x, f, r=16, residual=False, downsample=False, name='SP_res_'):
    strides = (2,2) if downsample else (1,1)

    x = Conv2D(f, (3,3), strides=strides, activation=None)(_x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(f, (3,3), activation=None)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)





def SP_net(num_classes,
          input_shape,
          r=16):
          # ...
