"""Define the model."""

import tensorflow as tf
from keras.applications import VGG16
from keras.applications import densenet
from keras.applications import inception_resnet_v2
from keras.utils import to_categorical
from keras.utils import plot_model
from keras import models
from keras import layers
from keras import optimizers
from keras import regularizers
import os
import sys
import time
from time import localtime, strftime
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

ADDTNL_TBOARD_TEXT = 'preprocess_testing'
TENSORBOARD_BASE_DIR = 'experiments/tensorboard'


def build_model(is_training, params):
    height, width, channel = params['height'], params['width'], params['channel']
    x = layers.Input(shape=(height, width, channel), name='main_input')

    # inception_res_net = inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(height, width, channel), pooling=None)(x)

    dense_net = densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(
        height, width, channel), pooling=None)(x)

    # vgg_net = VGG16(weights='imagenet', include_top=False)
    # vgg_net.trainable = True
    # set_trainable = False
    # for layer in vgg_net.layers:
    #     if layer.name == 'block5_conv1':
    #         set_trainable = True
    #     layer.trainable = set_trainable
    # vgg_net = vgg_net(x)

    # this code takes VGG16, and then add on a lauer of softmax to classify stuff.
    flatten = layers.Flatten()(dense_net)
    dropout = layers.Dropout(0.5)(flatten)
    batch_norm = layers.normalization.BatchNormalization()(dropout)

    bin_stenosis = layers.Dense(64, activation='relu')(batch_norm)
    bin_stenosis = layers.Dropout(0.2)(bin_stenosis)
    bin_stenosis = layers.normalization.BatchNormalization()(bin_stenosis)
    bin_stenosis = layers.Dense(10, activation='relu')(bin_stenosis)
    bin_stenosis = layers.Dense(
        1, activation='sigmoid', name='stenosis_output')(bin_stenosis)

    anatomy = layers.Dense(20, activation='relu')(batch_norm)
    anatomy = layers.Dense(4, activation='softmax',
                           name='anatomy_output')(anatomy)

    model = models.Model(inputs=[x], outputs=[bin_stenosis,
                                              anatomy])
    print(model.summary())
    # plot_model(model, to_file='model.png')
    return model


def get_model_name(epochs):
    return "epochs_{}_{}_{}".format(
        epochs, ADDTNL_TBOARD_TEXT, strftime("%Y-%m-%d_%H-%M-%S", localtime()))


def train_model(model, train_labels_stenosis, train_labels_anatomy, train_data, val_labels_stenosis, val_labels_anatomy, val_data, epochs=1, batch_size=16):

    MODEL_FINAL_DIR = '{}{}{}'.format(
        'experiments/weights/', get_model_name(epochs), '_weights.final.hdf5')
    MODEL_CP_DIR = '{}{}{}'.format(
        'experiments/weights/', get_model_name(epochs), '_weights.chkpt.hdf5')

    INIT_LR = 0.001
    adam = optimizers.Adam(lr=INIT_LR)
    model.compile(optimizer=adam,
                  loss={'stenosis_output': 'binary_crossentropy',
                        'anatomy_output': 'categorical_crossentropy'},
                  loss_weights={'stenosis_output': 2., 'anatomy_output': 1.},
                  metrics=['accuracy'])
    tensorboard = TensorBoard(log_dir=os.path.join(
        TENSORBOARD_BASE_DIR, get_model_name(epochs)))
    checkpoint = ModelCheckpoint(
        MODEL_CP_DIR, monitor='val_stenosis_output_acc', verbose=1, save_best_only=True, mode='max')
    lr_reduce = ReduceLROnPlateau(
        monitor='val_stenosis_output_acc', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

    history = model.fit(
        {'main_input': train_data},
        [train_labels_stenosis, train_labels_anatomy],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_data, [val_labels_stenosis, val_labels_anatomy]), callbacks=[tensorboard, checkpoint, lr_reduce])

    # save the pareameter of the model
    model.save(MODEL_FINAL_DIR)

    return history


def train_model_with_generators(model, train_flow, val_flow, epochs=1, steps_per_epoch=50, validation_steps=50):

    MODEL_FINAL_DIR = '{}{}{}'.format(
        'experiments/weights/', get_model_name(epochs), '_weights.final.hdf5')
    MODEL_CP_DIR = '{}{}{}'.format(
        'experiments/weights/', get_model_name(epochs), '_weights.chkpt.hdf5')

    INIT_LR = 0.001
    adam = optimizers.Adam(lr=INIT_LR)
    model.compile(optimizer=adam,
                  loss={'stenosis_output': 'binary_crossentropy',
                        'anatomy_output': 'categorical_crossentropy'},
                  loss_weights={'stenosis_output': 2., 'anatomy_output': 1.},
                  metrics=['accuracy'])
    tensorboard = TensorBoard(log_dir=os.path.join(
        TENSORBOARD_BASE_DIR, get_model_name(epochs)))
    checkpoint = ModelCheckpoint(
        MODEL_CP_DIR, monitor='val_stenosis_output_acc', verbose=1, save_best_only=True, mode='max')
    lr_reduce = ReduceLROnPlateau(
        monitor='val_stenosis_output_acc', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

    history = model.fit_generator(train_flow, epochs=epochs, steps_per_epoch=steps_per_epoch,
                                  validation_data=val_flow, validation_steps=validation_steps, callbacks=[tensorboard, checkpoint, lr_reduce])

    # save the pareameter of the model
    model.save(MODEL_FINAL_DIR)

    return history
