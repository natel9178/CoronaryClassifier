"""Train the model"""

import argparse
import logging
import os
import random

import tensorflow as tf

from model.utils import Params, set_logger, save_dict_to_json
import numpy as np
from model.input import imageload, merge_labels
from model.modelutils import numparize, describe
from model.model import build_model, train_model
from model.evaluate import print_plot_keras_metrics
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

    train_labels, train_data = imageload(mypath_train)
    dev_labels, dev_data = imageload(mypath_dev)

    # numparize the array
    train_label_np, train_data_np = numparize(train_labels, train_data)
    dev_label_np, dev_data_np = numparize(dev_labels, dev_data)

    # Describe the data
    height, width, channel, train_size, dev_size, categories, category_num = describe(
        train_label_np, train_data_np, dev_label_np, dev_data_np)

    # connecting to Keras below

    # Data extraction Part 2: extract first digit data (Left main or LAD or LCx or RCA)
    train_labels1, train_data1 = imageload(
        mypath_train, filename_label_position=1, dimheight=64,  dimwidth=64)
    dev_labels1, dev_data1 = imageload(
        mypath_dev, filename_label_position=1, dimheight=64,  dimwidth=64)

    # numparize the array

    train_label_np1, train_data_np1 = numparize(train_labels1, train_data1)
    dev_label_np1, dev_data_np1 = numparize(dev_labels1, dev_data1)

    # Describe the data
    height1, width1, channel1, train_size1, dev_size1, categories1, category_num1 = describe(
        train_label_np1, train_data_np1, dev_label_np1, dev_data_np1)

    # flatten the image and ensure it can go into Keras properly
    # notice the name is different from train_data.
    train_images = train_data_np.reshape((train_size, height, width, channel))
    train_images = train_images.astype('float32') / 255
    dev_images = dev_data_np.reshape((dev_size, height, width, channel))
    dev_images = dev_images.astype('float32') / 255

    train_images1 = train_data_np1.reshape((train_size1, height1, width1,
                                            channel1))
    train_images1 = train_images1.astype('float32') / 255
    dev_images1 = dev_data_np1.reshape((dev_size1, height1, width1, channel1))
    dev_images1 = dev_images1.astype('float32') / 255

    # general categoricla variable
    # notice I am add "s" at train_label, to distinguish from earlier numpy.
    train_labels = to_categorical(train_label_np)
    dev_labels = to_categorical(dev_label_np)

    train_labels1 = to_categorical(train_label_np1)
    dev_labels1 = to_categorical(dev_label_np1)

    train_labels_merged = merge_labels(train_labels, train_labels1)
    dev_labels_merged = merge_labels(dev_labels, dev_labels1)

    sample_size = dev_labels_merged.shape[0]  # sample size

    for i in range(sample_size):
        print(train_labels_merged[i, :], ",",
              train_label_np[i], ",", train_label_np1[i])

    params = {}
    params['height'] = height
    params['width'] = width
    params['channel'] = channel

    model = build_model(True, params)
    history = train_model(model, train_labels_merged,
                          train_images, dev_labels_merged, dev_images)

    # print the graph of learning history for diagnostic purpose.
    print_plot_keras_metrics(history)
