from keras import applications

# callables for pretrained model fetching
keras_apps = {'vgg16'               : applications.vgg16.VGG16,
              'vgg19'               : applications.vgg19.VGG19,
              'resnet50'            : applications.resnet50.ResNet50,
              'xception'            : applications.xception.Xception,
              'mobilenet'           : applications.mobilenet.MobileNet,
              'mobilenetv2'         : applications.mobilenetv2.MobileNetV2,
              'densenet121'         : applications.densenet.DenseNet121,
              'densenet169'         : applications.densenet.DenseNet169,
              'densenet201'         : applications.densenet.DenseNet201,
              'nasnet_large'        : applications.nasnet.NASNetLarge,
              'nasnet_mobile'       : applications.nasnet.NASNetMobile,
              'inception_v3'        : applications.inception_v3.InceptionV3,
              'inception_resnet_v2' : applications.inception_resnet_v2.InceptionResNetV2}

