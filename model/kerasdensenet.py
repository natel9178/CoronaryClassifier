"""DenseNet models for Keras.

# Reference paper

- [Densely Connected Convolutional Networks]
  (https://arxiv.org/abs/1608.06993) (CVPR 2017 Best Paper Award)

# Reference implementation

- [Torch DenseNets]
  (https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua)
- [TensorNets]
  (https://github.com/taehoonlee/tensornets/blob/master/tensornets/densenets.py)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from keras_applications import get_keras_submodule

backend = get_keras_submodule('backend')
engine = get_keras_submodule('engine')
layers = get_keras_submodule('layers')
models = get_keras_submodule('models')
keras_utils = get_keras_submodule('utils')

from keras_applications import imagenet_utils
from keras_applications.imagenet_utils import decode_predictions
from keras_applications.imagenet_utils import _obtain_input_shape


BASE_WEIGTHS_PATH = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.8/')
DENSENET121_WEIGHT_PATH = (
    BASE_WEIGTHS_PATH +
    'densenet121_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET121_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH +
    'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')
DENSENET169_WEIGHT_PATH = (
    BASE_WEIGTHS_PATH +
    'densenet169_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET169_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH +
    'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5')
DENSENET201_WEIGHT_PATH = (
    BASE_WEIGTHS_PATH +
    'densenet201_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET201_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH +
    'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5')


def dense_block(x, blocks, name):
    """A dense block.

    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """A transition block.

    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        Output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def DenseNet(blocks,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000):
    """Instantiates the DenseNet architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        blocks: numbers of building blocks for the four dense layers.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=221,
                                      data_format=backend.image_data_format(),
                                      require_flatten=False,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape, name='main_input')
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(
                tensor=input_tensor, shape=input_shape, name='main_input')
        else:
            img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')

    # Stenosis block
    bin_stenosis = transition_block(x, 0.5, name='pool4_stenosis')
    bin_stenosis = dense_block(bin_stenosis, blocks[3], name='conv5_stenosis')

    bin_stenosis = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn_stenosis')(bin_stenosis)

    if pooling == 'avg':
        bin_stenosis = layers.GlobalAveragePooling2D(
            name='avg_pool_stenosis')(bin_stenosis)
    elif pooling == 'max':
        bin_stenosis = layers.GlobalMaxPooling2D(
            name='max_pool_stenosis')(bin_stenosis)
    bin_stenosis = layers.BatchNormalization()(bin_stenosis)
    bin_stenosis = layers.Dense(40, activation='relu')(bin_stenosis)
    bin_stenosis = layers.BatchNormalization()(bin_stenosis)
    bin_stenosis = layers.Dense(40, activation='relu')(bin_stenosis)
    bin_stenosis = layers.Dense(
        1, activation='sigmoid', name='stenosis_output')(bin_stenosis)

    # Anatomy block
    anatomy = transition_block(x, 0.5, name='pool4_anatomy')
    anatomy = dense_block(anatomy, blocks[3], name='conv5_anatomy')

    anatomy = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn_anatomy')(anatomy)

    if pooling == 'avg':
        anatomy = layers.GlobalAveragePooling2D(
            name='avg_pool_anatomy')(anatomy)
    elif pooling == 'max':
        anatomy = layers.GlobalMaxPooling2D(name='max_pool_anatomy')(anatomy)
    anatomy = layers.BatchNormalization()(anatomy)
    anatomy = layers.Dense(20, activation='relu')(anatomy)
    anatomy = layers.Dense(4, activation='softmax',
                           name='anatomy_output')(anatomy)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = engine.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    # if blocks == [6, 12, 24, 16]:
    #     model = models.Model(inputs, x, name='densenet121')
    # elif blocks == [6, 12, 32, 32]:
    #     model = models.Model(inputs, x, name='densenet169')
    # elif blocks == [6, 12, 48, 32]:
    #     model = models.Model(inputs, x, name='densenet201')
    # else:
    #     model = models.Model(inputs, x, name='densenet')

    model = models.Model(inputs=inputs, outputs=[bin_stenosis,
                                                 anatomy])

    # Load weights.
    if weights == 'imagenet':
        if blocks == [6, 12, 24, 16]:
            weights_path = keras_utils.get_file(
                'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
                DENSENET121_WEIGHT_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='4912a53fbd2a69346e7f2c0b5ec8c6d3')
        elif blocks == [6, 12, 32, 32]:
            weights_path = keras_utils.get_file(
                'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5',
                DENSENET169_WEIGHT_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='50662582284e4cf834ce40ab4dfa58c6')
        elif blocks == [6, 12, 48, 32]:
            weights_path = keras_utils.get_file(
                'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5',
                DENSENET201_WEIGHT_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='1c2de60ee40562448dbac34a0737e798')
        model.load_weights(weights_path, by_name=True)
    elif weights is not None:
        model.load_weights(weights, by_name=True)

    return model


def DenseNet121(weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
    return DenseNet([6, 12, 24, 16], weights,
                    input_tensor, input_shape,
                    pooling, classes)


def DenseNet169(weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
    return DenseNet([6, 12, 32, 32], weights,
                    input_tensor, input_shape,
                    pooling, classes)


def DenseNet201(weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
    return DenseNet([6, 12, 48, 32], weights,
                    input_tensor, input_shape,
                    pooling, classes)


def preprocess_input(x, data_format=None):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 3D or 4D numpy array consists of RGB values within [0, 255].
        data_format: data format of the image tensor.

    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, data_format, mode='torch')


setattr(DenseNet121, '__doc__', DenseNet.__doc__)
setattr(DenseNet169, '__doc__', DenseNet.__doc__)
setattr(DenseNet201, '__doc__', DenseNet.__doc__)
