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
    x = layers.Input(shape=(height, width, channel), name='main_input')

    dense_net = densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(
        height, width, channel), pooling=None)(x)

    # this code takes VGG16, and then add on a lauer of softmax to classify stuff.
    flatten = layers.Flatten()(dense_net)
    bin_stenosis = layers.Dense(10, activation='relu')(flatten)
    bin_stenosis = layers.Dense(
        1, activation='sigmoid', name='stenosis_output')(bin_stenosis)

    anatomy = layers.Dense(20, activation='relu')(flatten)
    anatomy = layers.Dense(4, activation='softmax',
                           name='anatomy_output')(anatomy)

    model = models.Model(inputs=[x], outputs=[bin_stenosis,
                                              anatomy])
    print(model.summary())
    return model


def train_model(model, train_labels_stenosis, train_labels_anatomy, train_data, val_labels_stenosis, val_labels_anatomy, val_data, epochs=30, batch_size=16):
    INIT_LR = 0.0001
    adam = optimizers.Adam(lr=INIT_LR)
    model.compile(optimizer=adam, loss={'stenosis_output': 'binary_crossentropy', 'anatomy_output': 'categorical_crossentropy'},
                  metrics=['accuracy'])
    history = model.fit(
        {'main_input': train_data},
        {'stenosis_output': train_labels_stenosis,
            'anatomy_output': train_labels_anatomy},
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_data, {'stenosis_output': val_labels_stenosis, 'anatomy_output': val_labels_anatomy}))

    # save the pareameter of the model
    model.save('PL_CV_engine2.h5')

    return history
