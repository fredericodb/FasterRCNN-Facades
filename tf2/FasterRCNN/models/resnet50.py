#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# tf2/FasterRCNN/models/vgg16.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# TensorFlow/Keras implementation of the VGG-16 backbone for use as a feature
# extractor in Faster R-CNN. Only the convolutional layers are used.
#

from tkinter import Y
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import models
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Add
from tensorflow.keras.initializers import glorot_normal

global act
global add

class IdentityBlock(tf.keras.Model):

    def __init__(self, kernel_size, filters, stage, block, trainable=True):
        super(IdentityBlock, self).__init__(name='')
        global act
        global add

        nb_filter1, nb_filter2, nb_filter3 = filters
        
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        self.conv2a = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', trainable=trainable)
        self.bn2a = BatchNormalization(axis=3, name=bn_name_base + '2a')
        act += 1
        self.act2a = Activation('relu', name='activation_' + str(act))

        self.conv2b = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable)
        self.bn2b = BatchNormalization(axis=3, name=bn_name_base + '2b')
        act += 1
        self.act2b = Activation('relu', name='activation_' + str(act))

        self.conv2c = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)
        self.bn2c = BatchNormalization(axis=3, name=bn_name_base + '2c')

        add += 1
        self.merge = Add(name='merge_' + str(add))
        act += 1
        self.actmerge = Activation('relu', name='activation_' + str(act))

    def call(self, input_tensor):
        y = self.conv2a(input_tensor)
        y = self.bn2a(y)
        y = self.act2a(y)

        y = self.conv2b(y)
        y = self.bn2b(y)
        y = self.act2b(y)

        y = self.conv2c(y)
        y = self.bn2c(y)
        
        y = self.merge([y, input_tensor])
        y = self.actmerge(y)

        return y

class ConvBlock(tf.keras.Model):

    def __init__(self, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):
        super(ConvBlock, self).__init__(name='')
        global act
        global add

        nb_filter1, nb_filter2, nb_filter3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        self.conv2a = Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', trainable=trainable)
        self.bn2a = BatchNormalization(axis=3, name=bn_name_base + '2a')
        act += 1
        self.act2a = Activation('relu', name='activation_' + str(act))

        self.conv2b = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable)
        self.bn2b = BatchNormalization(axis=3, name=bn_name_base + '2b')
        act += 1
        self.act2b = Activation('relu', name='activation_' + str(act))

        self.conv2c = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)
        self.bn2c = BatchNormalization(axis=3, name=bn_name_base + '2c')

        self.shortcutconv = Conv2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', trainable=trainable)
        self.shortcutbn = BatchNormalization(axis=3, name=bn_name_base + '1')

        add += 1
        self.merge = Add(name='merge_' + str(add))
        act += 1
        self.actmerge = Activation('relu', name='activation_' + str(act))

    def call(self, input_tensor):
        y = self.conv2a(input_tensor)
        y = self.bn2a(y)
        y = self.act2a(y)

        y = self.conv2b(y)
        y = self.bn2b(y)
        y = self.act2b(y)

        y = self.conv2c(y)
        y = self.bn2c(y)

        z = self.shortcutconv(input_tensor)
        z = self.shortcutbn(y)

        y = self.merge([y, z])
        y = self.act2b(y)

        return y
    

class FeatureExtractor(tf.keras.Model):
  def __init__(self, l2 = 0):
    super().__init__()
    global act
    global add

    initial_weights = glorot_normal()
    regularizer = tf.keras.regularizers.l2(l2)
    input_shape = (None, None, 3)

    # First two convolutional blocks are frozen (not trainable)
    trainable = False
    self._zeropadding = ZeroPadding2D(name='zeropadding2d_1', input_shape=input_shape, padding=(3, 3))
    self._conv1 = Conv2D(name='conv1', filters = 64, kernel_size = (7, 7), strides = 2, trainable = trainable)
    self._bn_conv1 = BatchNormalization(axis=3, name='bn_conv1')
    act = 1
    self._activation1 = Activation('relu', name='activation_' + str(act))
    self._maxpooling2d_1 = MaxPooling2D((3, 3), strides=(2, 2), name='maxpooling2d_1')
    add = 0
    self.convblock2a = ConvBlock(3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable = trainable)
    self.identityblock2b = IdentityBlock(3, [64, 64, 256], stage=2, block='b', trainable = trainable)
    self.identityblock2c = IdentityBlock(3, [64, 64, 256], stage=2, block='c', trainable = trainable)

    # Weight decay begins from these layers onward: https://github.com/rbgirshick/py-faster-rcnn/blob/master/models/pascal_voc/VGG16/faster_rcnn_end2end/train.prototxt
    trainable = True
    self.convblock3a = ConvBlock(3, [128, 128, 512], stage=3, block='a', trainable = trainable)
    self.identityblock3b = IdentityBlock(3, [128, 128, 512], stage=3, block='b', trainable = trainable)
    self.identityblock3c = IdentityBlock(3, [128, 128, 512], stage=3, block='c', trainable = trainable)
    self.identityblock3d = IdentityBlock(3, [128, 128, 512], stage=3, block='d', trainable = trainable)

    self.convblock4a = ConvBlock(3, [256, 256, 1024], stage=4, block='a', trainable = trainable)
    self.identityblock4b = IdentityBlock(3, [256, 256, 1024], stage=4, block='b', trainable = trainable)
    self.identityblock4c = IdentityBlock(3, [256, 256, 1024], stage=4, block='c', trainable = trainable)
    self.identityblock4d = IdentityBlock(3, [256, 256, 1024], stage=4, block='d', trainable = trainable)
    self.identityblock4e = IdentityBlock(3, [256, 256, 1024], stage=4, block='e', trainable = trainable)
    self.identityblock4f = IdentityBlock(3, [256, 256, 1024], stage=4, block='f', trainable = trainable)

    self.convblock5a = ConvBlock(3, [512, 512, 2048], stage=5, block='a', trainable = trainable)
    self.identityblock5b = IdentityBlock(3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
    self.identityblock5c = IdentityBlock(3, [512, 512, 2048], stage=5, block='c', trainable=trainable)

    self.avgpooling2d = AveragePooling2D(pool_size=(2, 2), padding='same')

  def call(self, input_image):
    y = self._zeropadding(input_image)
    y = self._conv1(y)
    y = self._bn_conv1(y)
    y = self._activation1(y)
    y = self._maxpooling2d_1(y)

    y = self.convblock2a(y)
    y = self.identityblock2b(y)
    y = self.identityblock2c(y)

    y = self.convblock3a(y)
    y = self.identityblock3b(y)
    y = self.identityblock3c(y)
    y = self.identityblock3d(y)

    y = self.convblock4a(y)
    y = self.identityblock4b(y)
    y = self.identityblock4c(y)
    y = self.identityblock4d(y)
    y = self.identityblock4e(y)
    y = self.identityblock4f(y)

    y = self.convblock5a(y)
    y = self.identityblock5b(y)
    y = self.identityblock5c(y)

    y = self.avgpooling2d(y)

    return y
