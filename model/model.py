"""Define the model."""

import tensorflow as tf
from keras.applications import VGG16
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras import optimizers


def build_model(is_training, params):
    height, width, channel = params['height'], params['width'], params['channel']
    conv_base = VGG16(weights='imagenet', include_top=False,
                      input_shape=(height, width, channel))
    print(conv_base.summary())

    # this code takes VGG16, and then add on a lauer of softmax to classify stuff.

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(5, activation='sigmoid'))

    # Below freeze all the the VGG16 as untrainable except the last few layers. Look at structure of the VGG16 listed above

    conv_base.trainable = True
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

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
        epochs=30,
        batch_size=16,
        validation_data=(val_data, val_labels))

    # save the pareameter of the model
    model.save('PL_CV_engine2.h5')

    return history
