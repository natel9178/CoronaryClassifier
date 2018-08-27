"""Define the model."""

import tensorflow as tf
from keras.applications import VGG16
from keras.applications import densenet
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras import optimizers
from keras import regularizers


def build_model(is_training, params):
    height, width, channel = params['height'], params['width'], params['channel']
    # conv_base = VGG16(weights='imagenet', include_top=False,
    #                   input_shape=(height, width, channel))

    dense_net = densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(
        height, width, channel), pooling=None)

    # this code takes VGG16, and then add on a lauer of softmax to classify stuff.

    model = models.Sequential()
    model.add(dense_net)
    model.add(layers.Flatten())
    model.add(layers.Dense(5, activation='sigmoid'))

    # Below freeze all the the VGG16 as untrainable except the last few layers. Look at structure of the VGG16 listed above

    # conv_base.trainable=True
    # set_trainable=False
    # for layer in conv_base.layers:
    #     if layer.name == 'block5_conv1':
    #         set_trainable=True
    #     layer.trainable=set_trainable
    print(model.summary())
    return model


def train_model(model, train_labels, train_data, val_labels, val_data, epochs=30, batch_size=16):
    INIT_LR = 0.0001
    adam = optimizers.Adam(lr=INIT_LR)
    model.compile(optimizer=adam, loss='binary_crossentropy',
                  metrics=['accuracy'])
    print("dimension of train_images, train_labels,dev_images, dev_labels")
    print(train_data.shape, train_labels.shape, val_data.shape,
          val_labels.shape)
    history = model.fit(
        train_data,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_data, val_labels))

    # save the pareameter of the model
    model.save('PL_CV_engine2.h5')

    return history
