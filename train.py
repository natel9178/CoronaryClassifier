"""Train the model"""

import argparse
import logging
import os
import random

import tensorflow as tf

from model.utils import Params, set_logger, save_dict_to_json
import numpy as np
from model.input import imageload, merge_labels, expose_generators, generator_hotfix
from model.modelutils import numparize, describe
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
    train_label_np_stenosis, train_data_np = numparize(
        train_labels_stenosis, train_data)
    dev_label_np_stenosis, dev_data_np = numparize(
        dev_labels_stenosis, dev_data)

    # Describe the data
    height, width, channel, train_size, dev_size, categories, category_num = describe(
        train_label_np_stenosis, train_data_np, dev_label_np_stenosis, dev_data_np)

    # connecting to Keras below

    # Data extraction Part 2: extract first digit data (Left main or LAD or LCx or RCA)
    train_labels_anatomy, _ = imageload(
        mypath_train, filename_label_position=1)
    dev_labels_anatomy, _ = imageload(
        mypath_dev, filename_label_position=1)

    # numparize the array

    train_label_np_anatomy, _ = numparize(
        train_labels_anatomy, train_data)
    dev_label_np_anatomy, _ = numparize(dev_labels_anatomy, dev_data)
    train_labels_anatomy_cat = to_categorical(train_label_np_anatomy)
    dev_labels_anatomy_cat = to_categorical(dev_label_np_anatomy)

    # Describe the data
    height1, width1, channel1, train_size1, dev_size1, categories1, category_num1 = describe(
        train_label_np_anatomy, train_data_np, dev_label_np_anatomy, dev_data_np)

    # flatten the image and ensure it can go into Keras properly
    # notice the name is different from train_data.
    train_images = train_data_np.reshape((train_size, height, width, channel))
    train_images = train_images.astype('float32') / 255
    dev_images = dev_data_np.reshape((dev_size, height, width, channel))
    dev_images = dev_images.astype('float32') / 255

    sample_size = dev_images.shape[0]  # sample size

    # for i in range(sample_size):
    #     print(dev_label_np_stenosis[i], ",", dev_labels_anatomy_cat[i])

    train_flow, val_flow = expose_generators(
        train_images, train_label_np_stenosis, train_labels_anatomy_cat, dev_images, dev_label_np_stenosis, dev_labels_anatomy_cat)

    train_flow_hf = generator_hotfix(train_flow)
    val_flow_hf = generator_hotfix(val_flow)

    params = {}
    params['height'] = height
    params['width'] = width
    params['channel'] = channel
    model = build_model(is_training=True, params=params)
    # history = train_model(model, train_label_np_stenosis, train_labels_anatomy_cat, train_images, dev_label_np_stenosis, dev_labels_anatomy_cat, dev_images)
    history = train_model_with_generators(
        model, train_flow_hf, val_flow_hf, epochs=1)

    # print the graph of learning history for diagnostic purpose.
    # print_plot_keras_metrics(history)
    # eval_model(model, dev_labels_stenosis, dev_data, dev_label_np_stenosis,
    #            dev_data_np, dev_label_np_anatomy, dev_data_np1)
