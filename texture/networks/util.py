import random
from tensorflow.keras import applications
from tensorflow.keras.layers import Dense, Dropout, Lambda, Conv2D
from tensorflow.keras.models import Model as KerasModel

from texture.layers import Encoding, KernelPooling
from texture.ops import bilinear_pooling

# Callables for ImageNet-pretrained model fetching
keras_apps = {'vgg16'               : applications.vgg16.VGG16,
              'vgg19'               : applications.vgg19.VGG19,
              'resnet50'            : applications.resnet50.ResNet50,
              'xception'            : applications.xception.Xception,
              'mobilenet'           : applications.mobilenet.MobileNet,         # 'mobilenetv2'         : applications.mobilenetv2.MobileNetV2,
              'densenet121'         : applications.densenet.DenseNet121,
              'densenet169'         : applications.densenet.DenseNet169,
              'densenet201'         : applications.densenet.DenseNet201,
              'nasnet_large'        : applications.nasnet.NASNetLarge,
              'nasnet_mobile'       : applications.nasnet.NASNetMobile,
              'inception_v3'        : applications.inception_v3.InceptionV3,
              'inception_resnet_v2' : applications.inception_resnet_v2.InceptionResNetV2}


def make_backbone(backbone_cnn, input_shape):
    '''Check an existing backbone Model or grab ImageNet pretrained from keras_apps.'''
    if backbone_cnn is None:
        return None
    elif isinstance(backbone_cnn, KerasModel):
        assert len(backbone_cnn.output_shape)==4, 'backbone_cnn.output must output a 4D Tensor'
        return backbone_cnn
    elif isinstance(backbone_cnn, str):
        assert backbone_cnn in keras_apps.keys(), 'Invalid keras.applications string'
        model = keras_apps[backbone_cnn](include_top=False, input_shape=input_shape)
        # resnet50 ends with a 7x7 pooling, which collapses conv to 1x1 for 224x224 input
        if backbone_cnn == 'resnet50':
            model = KerasModel(inputs=model.input, outputs=model.layers[-2].output)
        return model
    else:
        raise ValueError('input to make_backbone() has invalid type')


def make_dense_layers(dense_layers, dropout=None):
    '''Instantiate a series of Dense layers, optionally with Dropout.'''
    if len(dense_layers) == 0:
        if dropout is not None:
            return lambda x: Dropout(rate=dropout)(x)
        else:
            return lambda x: x
    else:
        def dense_layers_fn(x):
            for N in dense_layers:
                x = Dense(N, activation='relu')(x)
                if dropout is not None:
                    x = Dropout(rate=dropout)(x)
            return x
        return dense_layers_fn


def make_pooling_layer(pooling_name, **kwargs):
    """Make a pooling layer with optional `conv1x1` reduction."""
    name_tail = str(kwargs.pop('name', random.randint(0, 2**16)))

    conv1x1 = kwargs.pop('conv1x1', None)
    if conv1x1 is not None:
        reducer_name = 'reduce_' + pooling_name + name_tail
        reducer = lambda x: Conv2D(conv1x1, (1,1), activation='relu', name=reducer_name)(x)
    else:
        reducer = lambda x: x

    pooler_name = pooling_name + '_' + name_tail
    if 'bilinear' in pooling_name.lower():
        pooler = lambda x: Lambda(bilinear_pooling, name=pooler_name)([x, x])
    elif 'encoding' in pooling_name.lower():
        pooler = lambda x: Encoding(**kwargs, name=pooler_name)(x)
    elif 'kernel' in pooling_name.lower():
        pooler = lambda x: KernelPooling(**kwargs, name=pooler_name)(x)
    else:
        raise RuntimeWarning('Unrecognized `pooling_name`: {}`'.format(pooling_name))
        pooler = lambda x: x

    return lambda x: pooler(reducer(x))


def auxiliary_pooling(x, pooling_name, output_size=256, dropout=None, **kwargs):
    """Aux. pooling branch with `output_size` dense features."""
    x = make_pooling_layer(pooling_name, **kwargs)(x)
    x = make_dense_layers([output_size], dropout=dropout)(x)
    return x
