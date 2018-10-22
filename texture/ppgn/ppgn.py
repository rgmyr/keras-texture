from tensorflow.keras.applications import vgg16



class PPGN():
    """

    """
    def __init__(self, ):

        self.E = encoder
        self.G = gan
        self.C = classifier

    def fit():



def make_encoder(input_shape, h1='block5_pool', h='fc1'):
    vgg_model = vgg16.VGG16(input_shape=input_shape, include_top=True)
    vgg_model.summary()


if __name__ == '__main__':
    make_encoder((224,224,3))
