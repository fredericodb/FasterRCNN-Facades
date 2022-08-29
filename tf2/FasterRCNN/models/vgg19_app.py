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
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.initializers import glorot_normal


class FeatureExtractor(tf.keras.Model):
  def __init__(self, l2 = 0):
    super().__init__()

    initial_weights = glorot_normal()
    regularizer = tf.keras.regularizers.l2(l2)
    input_shape = (None, None, 3)
    self._feature_extractor = tf.keras.applications.VGG19(input_shape=input_shape, include_top=False)
    self._feature_extractor.trainable = False

  def call(self, input_image):
    y = self._feature_extractor(input_image)

    return y
