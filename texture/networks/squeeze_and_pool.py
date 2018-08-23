
import tensorflow as tf
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Conv2D, BatchNormalization, Concatenate, MaxPool2D, Input, Reshape
from tensorflow.keras.layers import Lambda, Add, Multiply, Dense, Activation, GlobalAveragePooling2D

#from texture.initializers import ConvolutionAware
from texture.networks.util import make_dense_layers
from texture.ops import bilinear_pooling


def SE_retain_z(x, ratio, block_name='', dropout_rate=None):

    filters = x.get_shape().as_list()[-1]

    # squeeze
    z = GlobalAveragePooling2D(name=block_name+'_globalpool')(x)    # reshape?
    z = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False, name=block_name+"_compress")(z)

    # expand
    s = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False, name=block_name+'_expand')(z)

    # excite
    u = Multiply(name=block_name+'_excite')([x, s])

    return u, Reshape((1, 1, filters // ratio))(z)


def SP_block(_x, f, stage, ratio=16, residual=False, downsample=False, name='sp_res_block'):
    '''Using pre-activation version.'''

    block_name = name + str(stage)

    init = _x
    init_filters = init.get_shape().as_list()[-1]
    strides = (2,2) if downsample else (1,1)

    if residual and (downsample or init_filters != f):
        init = Conv2D(f, (1,1), strides=strides, kernel_initializer='he_normal', use_bias=False)(init)

    x = Conv2D(f, (3,3), strides=strides, padding='same', activation=None, name=block_name+'_conv1')(_x)
    x = BatchNormalization(name=block_name+'_bn1')(x)
    x = Activation('relu', name=block_name+'_relu1')(x)

    x = Conv2D(f, (3,3), padding='same', activation=None, name=block_name+'_conv2')(x)
    x = BatchNormalization(name=block_name+'_bn2')(x)
    x = Activation('relu', name=block_name+'_relu2')(x)

    # squeeze and get compressed vector
    if ratio is not None:
        x, z = SE_retain_z(x, ratio, block_name=block_name)
    else:
        z = None

    if residual:
        x = Add(name=block_name+'_shortcut')([x, init])

    return x, z


def SP_ResNet(num_classes,
              input_shape,
              depths=[2, 2, 2, 2],
              filters=[64, 128, 256, 512],
              pool_at=[0, 1, 2, 3],
              squeeze_ratio=16,
              dense_layers=[],
              dropout_rate=None):
    # ...

    input_img = Input(shape=input_shape, name='input')

    # entry conv + pool
    x = Conv2D(filters[0], (7,7), strides=(2,2), padding='same', activation=None, name='entry_conv')(input_img)
    x = BatchNormalization(name='entry_bn')(x)
    x = Activation('relu', name='entry_relu')(x)
    x = MaxPool2D((3,3), strides=(2,2), padding='same')(x)

    pooling_outputs = []
    for i, (f, d) in enumerate(zip(filters, depths)):
        # n_blocks = depth
        for n in range(d):
            downsample = True if n == 0 else False
            x, z = SP_block(x, f, str(i)+'_'+str(n), ratio=squeeze_ratio, residual=True, downsample=downsample)

        # only pool at last block in depth
        if i in pool_at and z is not None:
            z = Lambda(bilinear_pooling, name='bilinear_pooling'+str(i))([z, z])
            pooling_outputs.append(z)
            print(z.get_shape().as_list())

    x = GlobalAveragePooling2D(name='global_pooling_top')(x)
    pooling_outputs.append(x)
    x = Concatenate(name='feature_concat')(pooling_outputs)

    x = make_dense_layers(dense_layers, dropout=dropout_rate)(x)

    pred = Dense(num_classes, activation='softmax')(x)

    model = KerasModel(inputs=input_img, outputs=pred)

    return model
