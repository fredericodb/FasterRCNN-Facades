#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# tf2/FasterRCNN/models/vgg16.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# TensorFlow/Keras implementation of the VGG-16 backbone for use as a feature
# extractor in Faster R-CNN. Only the convolutional layers are used.
#

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.initializers import glorot_normal

from . import vgg16
from . import vgg19
from . import resnet50
# from . import resnet101
# from . import resnet152

def BackboneModel(name, input_shape=None, include_top=True):
  if   name == 'vgg16':
    return tf.keras.applications.VGG16(input_shape=input_shape, include_top=include_top, weights = "imagenet")
  elif name == 'vgg19':
    return tf.keras.applications.VGG19(input_shape=input_shape, include_top=include_top, weights = "imagenet")
  elif name == 'resnet50':
    return tf.keras.applications.ResNet50(input_shape=input_shape, include_top=include_top, weights = "imagenet")
  elif name == 'resnet101':
    return tf.keras.applications.ResNet101(input_shape=input_shape, include_top=include_top, weights = "imagenet")
  elif name == 'resnet152':
    return tf.keras.applications.ResNet152(input_shape=input_shape, include_top=include_top, weights = "imagenet")

def BackboneFlatten(name, y):
  if   name == 'vgg16':
    return y
  elif name == 'vgg19':
    return y
  elif name == 'resnet50':
    return tf.keras.layers.Flatten()(y)
  elif name == 'resnet101':
    return tf.keras.layers.Flatten()(y)
  elif name == 'resnet152':
    return tf.keras.layers.Flatten()(y)


class FeatureExtractor(tf.keras.Model):
  def __init__(self, backbone = 'vgg16', l2 = 0):
    super().__init__()

    initial_weights = glorot_normal()
    regularizer = tf.keras.regularizers.l2(l2)
    input_shape = (None, None, 3)
    self._backbone = backbone
    self._feature_extractor = BackboneModel(name=backbone, input_shape=input_shape, include_top=False)
    self._feature_extractor.trainable = False
    self._layers = []
    for layer in self._feature_extractor.layers:
      self._layers.append(layer)
    self._layers = self._layers[:-1]


  def call(self, input_image):
    first = True
    for layer in self._layers:
      if first:
        y = layer(input_image)
        first = False
      else:
        y = layer(y)
    y = BackboneFlatten(self._backbone, y)

    return y
