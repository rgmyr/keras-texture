
from tensorflow.keras import applications
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model as KerasModel

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
    if isinstance(backbone_cnn, KerasModel):
        assert len(backbone_cnn.output_shape)==4, 'backbone_cnn.output must output a 4D Tensor'
        return backbone_cnn
    elif isinstance(backbone_cnn, str):
        assert backbone_cnn in keras_apps.keys(), 'Invalid keras.applications string'
        return keras_apps[backbone_cnn](include_top=False, input_shape=input_shape)
    else:
        raise ValueError('input to make_backbone() has invalid type')


def make_dense_layers(dense_layers, dropout=None):
    '''Instantiate a series of Dense layers, optionally with Dropout.'''
    def dense_layers(x):
        for N in dense_layers:
            x = Dense(N, activation='relu')(x)
            if dropout is not None:
                x = Dropout(rate=dropout)(x)
        return x
    return dense_layers
