#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
#

# from . import vgg16
# from . import vgg19
# from . import resnet50
# from . import resnet101
# from . import resnet152
import tensorflow as tf
import tensorflow.keras.applications as backbone
from tensorflow.keras import Model
from tensorflow.keras.initializers import glorot_normal
import os
import tempfile


def Predictor(name, weights='imagenet'):
    if name == 'vgg16':
        return backbone.VGG16(weights=weights, include_top=False)
    elif name == 'vgg19':
        return backbone.VGG19(weights=weights, include_top=False)
    elif name == 'resnet50':
        return backbone.ResNet50(weights=weights, include_top=False)
    elif name == 'resnet101':
        return backbone.ResNet101(weights=weights, include_top=False)
    elif name == 'resnet152':
        return backbone.ResNet152(weights=weights, include_top=False)
    elif name == 'resnet50v2':
        return backbone.ResNet50V2(weights=weights, include_top=False)
    elif name == 'resnet101v2':
        return backbone.ResNet101V2(weights=weights, include_top=False)
    elif name == 'resnet152v2':
        return backbone.ResNet152V2(weights=weights, include_top=False)
    elif name == 'densenet121':
        return backbone.DenseNet121(weights=weights, include_top=False)
    elif name == 'densenet169':
        return backbone.DenseNet169(weights=weights, include_top=False)
    elif name == 'densenet201':
        return backbone.DenseNet201(weights=weights, include_top=False)
    elif name == 'inceptionresnetv2':
        return backbone.InceptionResNetV2(weights=weights, include_top=False)
    elif name == 'inceptionv3':
        return backbone.InceptionV3(weights=weights, include_top=False)
    elif name == 'mobilenet':
        return backbone.MobileNet(weights=weights, include_top=False)
    elif name == 'mobilenetv2':
        return backbone.MobileNetV2(weights=weights, include_top=False)
    elif name == 'nasnetlarge':
        return backbone.NASNetLarge(weights=weights, include_top=False)
    elif name == 'nasnetmobile':
        return backbone.NASNetMobile(weights=weights, include_top=False)
    elif name == 'xception':
        return backbone.Xception(weights=weights, include_top=False)
    elif name == 'efficientnetb7':
        return backbone.EfficientNetB7(weights=weights, include_top=False)


def BackboneFeatureExtractor(name, l2=0, partially_trainable=True):
    def add_regularization(model, regularizer=tf.keras.regularizers.l2(0.0001)):

        if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
            print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
            return model

        for layer in model.layers:
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, regularizer)

        # When we change the layers attributes, the change only happens in the model config file
        model_json = model.to_json()

        # Save the weights before reloading the model.
        tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
        model.save_weights(tmp_weights_path)

        # load the model from the config
        model = tf.keras.models.model_from_json(model_json)

        # Reload the model weights
        model.load_weights(tmp_weights_path, by_name=True)
        return model

    initial_weights = glorot_normal()
    regularizer = tf.keras.regularizers.l2(l2)

    if name == 'vgg16':
        model = backbone.VGG16(weights=None, include_top=False)
        if not partially_trainable:
            model.trainable = False
        else:
            model.trainable = True
            model.get_layer('block1_conv1').trainable = False
            model.get_layer('block1_conv2').trainable = False
            model.get_layer('block2_conv1').trainable = False
            model.get_layer('block2_conv2').trainable = False
            model = add_regularization(model, regularizer=regularizer)
        return Model(inputs=model.input, outputs=model.layers[-2].output)  # bottleneck before block5_pool (7x7)
    elif name == 'vgg19':
        model = backbone.VGG19(weights=None, include_top=False)
        if not partially_trainable:
            model.trainable = False
        else:
            model.trainable = True
            model.get_layer('block1_conv1').trainable = False
            model.get_layer('block1_conv2').trainable = False
            model.get_layer('block2_conv1').trainable = False
            model.get_layer('block2_conv2').trainable = False
            model = add_regularization(model, regularizer=regularizer)
        return Model(inputs=model.input, outputs=model.layers[-2].output)  # bottleneck before block5_pool (7x7)
    elif name == 'resnet50':
        model = backbone.ResNet50(weights=None, include_top=False)
        if not partially_trainable:
            model.trainable = False
        else:
            model.trainable = False
            for layer in model.layers[144:]:
                layer.trainable = True
            model = add_regularization(model, regularizer=regularizer)
        return Model(inputs=model.input, outputs=model.layers[-1].output)  # bottleneck before avg_pool (1x1)
    elif name == 'resnet101':
        model = backbone.ResNet101(weights=None, include_top=False)
        if not partially_trainable:
            model.trainable = False
        else:
            model.trainable = False
            for layer in model.layers[313:]:
                layer.trainable = True
        return Model(inputs=model.input, outputs=model.layers[-1].output)  # bottleneck before avg_pool (1x1)
    elif name == 'resnet152':
        model = backbone.ResNet152(weights=None, include_top=False)
        if not partially_trainable:
            model.trainable = False
        else:
            model.trainable = False
            for layer in model.layers[483:]:
                layer.trainable = True
        return Model(inputs=model.input, outputs=model.layers[-1].output)  # bottleneck before avg_pool (1x1)
    elif name == 'resnet50v2':
        model = backbone.ResNet50V2(weights=None, include_top=False)
        if not partially_trainable:
            model.trainable = False
        else:
            model.trainable = False
            for layer in model.layers[144:]:
                layer.trainable = True
            model = add_regularization(model, regularizer=regularizer)
        return Model(inputs=model.input, outputs=model.layers[-1].output)
    elif name == 'resnet101v2':
        model = backbone.ResNet101V2(weights=None, include_top=False)
        if not partially_trainable:
            model.trainable = False
        else:
            model.trainable = False
            for layer in model.layers[313:]:
                layer.trainable = True
        return Model(inputs=model.input, outputs=model.layers[-1].output)
    elif name == 'resnet152v2':
        model = backbone.ResNet152V2(weights=None, include_top=False)
        if not partially_trainable:
            model.trainable = False
        else:
            model.trainable = False
            for layer in model.layers[483:]:
                layer.trainable = True
        return Model(inputs=model.input, outputs=model.layers[-1].output)
    elif name == 'densenet121':
        model = backbone.DenseNet121(weights=None, include_top=False)
        if not partially_trainable:
            model.trainable = False
        else:
            model.trainable = False
            for layer in model.layers[313:]:
                layer.trainable = True
        return Model(inputs=model.input, outputs=model.layers[-2].output)  # bottleneck before bn (7x7)
    elif name == 'densenet169':
        model = backbone.DenseNet169(weights=None, include_top=False)
        if not partially_trainable:
            model.trainable = False
        else:
            model.trainable = False
            for layer in model.layers[369:]:
                layer.trainable = True
        return Model(inputs=model.input, outputs=model.layers[-2].output)  # bottleneck before bn (7x7)
    elif name == 'densenet201':
        model = backbone.DenseNet201(weights=None, include_top=False)
        if not partially_trainable:
            model.trainable = False
        else:
            model.trainable = False
            for layer in model.layers[481:]:
                layer.trainable = True
        return Model(inputs=model.input, outputs=model.layers[-2].output)  # bottleneck before bn (7x7)
    elif name == 'inceptionresnetv2':
        model = backbone.InceptionResNetV2(weights=None, include_top=False)
        if not partially_trainable:
            model.trainable = False
        else:
            model.trainable = False
            for layer in model.layers[120:]:
                layer.trainable = True
        return Model(inputs=model.input, outputs=model.layers[-2].output)  # bottleneck before conv_7b_ac (8x8)
    elif name == 'inceptionv3':
        model = backbone.InceptionV3(weights=None, include_top=False)
        if not partially_trainable:
            model.trainable = False
        else:
            model.trainable = False
            for layer in model.layers[120:]:
                layer.trainable = True
        return Model(inputs=model.input, outputs=model.layers[-2].output)  # bottleneck before mixed10 (8x8)
    elif name == 'mobilenet':
        model = backbone.MobileNet(weights=None, include_top=False)
        if not partially_trainable:
            model.trainable = False
        else:
            model.trainable = False
            for layer in model.layers[73:]:
                layer.trainable = True
        return Model(inputs=model.input, outputs=model.layers[-2].output)  # bottleneck before conv_pw_13_relu (7x7)
    elif name == 'mobilenetv2':
        model = backbone.MobileNetV2(weights=None, include_top=False)
        if not partially_trainable:
            model.trainable = False
        else:
            model.trainable = False
            for layer in model.layers[120:]:
                layer.trainable = True
        return Model(inputs=model.input, outputs=model.layers[-2].output)  # bottleneck before out_relu (7x7)
    elif name == 'nasnetlarge':
        model = backbone.NASNetLarge(weights=None, include_top=False)
        if not partially_trainable:
            model.trainable = False
        else:
            model.trainable = False
            for layer in model.layers[120:]:
                layer.trainable = True
        return Model(inputs=model.input, outputs=model.layers[-2].output)  # bottleneck before activation_260 (11x11)
    elif name == 'nasnetmobile':
        model = backbone.NASNetMobile(weights=None, include_top=False)
        if not partially_trainable:
            model.trainable = False
        else:
            model.trainable = False
            for layer in model.layers[120:]:
                layer.trainable = True
        return Model(inputs=model.input, outputs=model.layers[-2].output)  # bottleneck before activation_188 (7x7)
    elif name == 'xception':
        model = backbone.Xception(weights=None, include_top=False)
        if not partially_trainable:
            model.trainable = False
        else:
            model.trainable = False
            for layer in model.layers[126:]:
                layer.trainable = True
        return Model(inputs=model.input, outputs=model.layers[-2].output)
    elif name == 'efficientnetb7':
        model = backbone.EfficientNetB7(weights=None, include_top=False)
        if not partially_trainable:
            model.trainable = False
        else:
            model.trainable = False
            for layer in model.layers[120:]:
                layer.trainable = True
        return Model(inputs=model.input, outputs=model.layers[-1].output)

def BackboneFeatureMapSize(name, width, height):
    def VGG16_output_size(input_length):
        return input_length // 16

    def VGG19_output_size(input_length):
        return input_length // 16

    def ResNet50_output_size(input_length):
        # zero_pad
        input_length += 6
        # apply 4 strided convolutions
        filter_sizes = [7, 3, 1, 1, 1]
        stride = 2
        for filter_size in filter_sizes:
            input_length = (input_length - filter_size + stride) // stride
        return input_length

    def ResNet101_output_size(input_length):
        # zero_pad
        input_length += 6
        # apply 4 strided convolutions
        filter_sizes = [7, 3, 1, 1, 1]
        stride = 2
        for filter_size in filter_sizes:
            input_length = (input_length - filter_size + stride) // stride
        return input_length

    def ResNet152_output_size(input_length):
        # zero_pad
        input_length += 6
        # apply 4 strided convolutions
        filter_sizes = [7, 3, 1, 1, 1]
        stride = 2
        for filter_size in filter_sizes:
            input_length = (input_length - filter_size + stride) // stride
        return input_length

    def MobileNet_output_size(input_length):
        return input_length // 32  # there is 4 strides.

    def MobileNetV2_output_size(input_length):
        # return input_length // 32 + 1  # there is 5 strides.
        return input_length // 32  # there is 5 strides.
        # return input_length // 16  # there is 5 strides.

    if name == 'vgg16':
        return VGG16_output_size(width), VGG16_output_size(height)
    elif name == 'vgg19':
        return VGG19_output_size(width), VGG19_output_size(height)
    elif name == 'resnet50':
        return ResNet50_output_size(width), ResNet50_output_size(height)
    elif name == 'resnet101':
        return ResNet101_output_size(width), ResNet101_output_size(height)
    elif name == 'resnet152':
        return ResNet152_output_size(width), ResNet152_output_size(height)
    elif name == 'mobilenet':
        return MobileNet_output_size(width), MobileNet_output_size(height)
    elif name == 'mobilenetv2':
        return MobileNetV2_output_size(width), MobileNetV2_output_size(height)


def BackboneFeaturePixels(name):
    if name == 'vgg16':
        return 16
    elif name == 'vgg19':
        return 16
    elif name == 'resnet50':
        return 31
    elif name == 'resnet101':
        return 31
    elif name == 'resnet152':
        return 31
    elif name == 'resnet50v2':
        return 31
    elif name == 'resnet101v2':
        return 31
    elif name == 'resnet152v2':
        return 31
    elif name == 'densenet121':
        return 32
    elif name == 'densenet169':
        return 32
    elif name == 'densenet201':
        return 32
    elif name == 'inceptionresnetv2':  # aspect ratio problem (17, 36) feature map
        return 33
    elif name == 'inceptionv3':  # aspect ratio problem (17, 36) feature map
        return 33
    elif name == 'mobilenet':
        return 32
    elif name == 'mobilenetv2':
        return 32
    elif name == 'nasnetlarge':
        return 32
    elif name == 'nasnetmobile':
        return 32
    elif name == 'xception':
        return 31
    elif name == 'efficientnetb7':
        return 32
