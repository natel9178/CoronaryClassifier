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

ADDTNL_TBOARD_TEXT = 'datav2_densev2_try1'
TENSORBOARD_BASE_DIR = 'experiments/tensorboard'


def build_model(is_training, params):
    height, width, channel = params['height'], params['width'], params['channel']
    x = layers.Input(shape=(height, width, channel), name='main_input')

    model = DenseNet121(weights='imagenet', input_shape=(
        height, width, channel), input_tensor=x, pooling='avg')
    # print(model.summary())
    # plot_model(model, to_file='model.png')
    return model


def get_current_time_string():
    return strftime("%Y-%m-%d_%H-%M-%S", localtime())


def get_model_name(epochs):
    return "epochs_{}_{}_{}".format(
        epochs, ADDTNL_TBOARD_TEXT, get_current_time_string())


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
        monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

    history = model.fit(
        {'main_input': train_data},
        [train_labels_stenosis, train_labels_anatomy],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_data, [val_labels_stenosis, val_labels_anatomy]), callbacks=[tensorboard, checkpoint, lr_reduce])

    # save the pareameter of the model
    model.save(MODEL_FINAL_DIR)

    return history


def train_model_with_generators(model, train_flow, val_flow, epochs=1, lr=0.001, steps_per_epoch=50, validation_steps=50, model_weight_filename=None, starting_epoch=0):
    COMMON_WEIGHT_DIR = 'experiments/weights/'
    MODEL_FINAL_DIR = '{}{}{}'.format(
        COMMON_WEIGHT_DIR, get_model_name(epochs), '_weights.final.hdf5')
    MODEL_CP_DIR = '{}{}{}'.format(
        COMMON_WEIGHT_DIR, get_model_name(epochs), '_weights.chkpt.hdf5')
    if model_weight_filename != None:
        MODEL_CP_DIR.replace('_weights.chkpt.hdf5', '')
        MODEL_CP_DIR = '{}{}{}{}'.format(
            MODEL_CP_DIR, '_resume_',  get_current_time_string(), '_weights.chkpt.hdf5')

    adam = optimizers.Adam(lr=lr)
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
        monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

    if model_weight_filename != None:
        model.load_weights(model_weight_filename, by_name=True)

    history = model.fit_generator(train_flow, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=val_flow,
                                  validation_steps=validation_steps, callbacks=[tensorboard, checkpoint, lr_reduce], initial_epoch=int(starting_epoch))

    # save the pareameter of the model
    model.save(MODEL_FINAL_DIR)

    return history
