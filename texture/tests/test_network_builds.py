import unittest
import numpy as np

from texture.models import TextureModel
from texture.networks import bilinear_cnn, deepten, deep_net, dilation_net, kernel_pooling_cnn

num_classes = 10
input_size = 224

class TestBCNN(unittest.TestCase):
    def test_build(self):
        bcnn = bilinear_cnn(10, 224, 'vgg16')
        self.assertIsNotNone(bcnn)
