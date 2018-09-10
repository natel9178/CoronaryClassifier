"""Train the model"""

import argparse
import logging
import os
import random

import tensorflow as tf

from model.utils import Params, set_logger, save_dict_to_json
import numpy as np
from model.input import imageload, merge_labels, expose_generators, generator_hotfix
from model.modelutils import numparize_data, numparize_labels, describe
from model.model import build_model, train_model, train_model_with_generators
from model.evaluate import print_plot_keras_metrics, eval_model
from keras.applications import VGG16
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras import optimizers


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")
parser.add_argument('--starting_epoch', default=0,
                    help="Optional, starting epoch for retrainint")
parser.add_argument('--learning_rate', default=0.001,
                    help="Optional, starting epoch for retrainint")


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)
    args = parser.parse_args()

    # Part 1: extract first digit data (Normal or abnormal)

    # define the paths to training, development sets
    mypath_train = os.path.join(args.data_dir, 'train')
    mypath_dev = os.path.join(args.data_dir, 'dev')

    train_labels_stenosis, train_data = imageload(mypath_train)
    dev_labels_stenosis, dev_data = imageload(mypath_dev)

    # numparize the array
    train_label_np_stenosis = numparize_labels(train_labels_stenosis)
    del train_labels_stenosis

    train_data_np = numparize_data(train_data)
    del train_data

    dev_label_np_stenosis = numparize_labels(dev_labels_stenosis)
    del dev_labels_stenosis

    dev_data_np = numparize_data(dev_data)
    del dev_data

    # Describe the data
    height, width, channel, train_size, dev_size, _, _ = describe(
        train_label_np_stenosis, train_data_np, dev_label_np_stenosis, dev_data_np)

    # connecting to Keras below

    # Data extraction Part 2: extract first digit data (Left main or LAD or LCx or RCA)
    train_labels_anatomy, _ = imageload(
        mypath_train, filename_label_position=1, ignore_image_data=True)
    dev_labels_anatomy, _ = imageload(
        mypath_dev, filename_label_position=1, ignore_image_data=True)

    # numparize the array

    train_label_np_anatomy = numparize_labels(train_labels_anatomy)
    dev_label_np_anatomy = numparize_labels(dev_labels_anatomy)
    train_labels_anatomy_cat = to_categorical(train_label_np_anatomy)
    dev_labels_anatomy_cat = to_categorical(dev_label_np_anatomy)

    # Describe the data
    describe(train_label_np_anatomy, train_data_np,
             dev_label_np_anatomy, dev_data_np)
    del train_label_np_anatomy
    del dev_label_np_anatomy

    # flatten the image and ensure it can go into Keras properly
    # notice the name is different from train_data.
    train_images = train_data_np.reshape((train_size, height, width, channel))
    dev_images = dev_data_np.reshape((dev_size, height, width, channel))

    sample_size = dev_images.shape[0]  # sample size

    BATCH_SIZE = 32
    train_flow, val_flow = expose_generators(
        train_images, train_label_np_stenosis, train_labels_anatomy_cat, dev_images, dev_label_np_stenosis, dev_labels_anatomy_cat, BATCH_SIZE)

    train_flow_hf = generator_hotfix(train_flow)
    val_flow_hf = generator_hotfix(val_flow)

    params = {}
    params['height'] = height
    params['width'] = width
    params['channel'] = channel
    model = build_model(is_training=True, params=params)
    # history = train_model(model, train_label_np_stenosis, train_labels_anatomy_cat, train_images, dev_label_np_stenosis, dev_labels_anatomy_cat, dev_images)
    history = train_model_with_generators(
        model, train_flow_hf, val_flow_hf, epochs=200, lr=float(args.learning_rate), steps_per_epoch=len(train_images)/BATCH_SIZE, validation_steps=len(dev_images)/BATCH_SIZE, model_weight_filename=args.restore_from, starting_epoch=args.starting_epoch)

    # print the graph of learning history for diagnostic purpose.
    # print_plot_keras_metrics(history)
    # eval_model(model, dev_labels_stenosis, dev_data, dev_label_np_stenosis,
    #            dev_data_np, dev_label_np_anatomy, dev_data_np1)
