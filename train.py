"""Train the model"""

import argparse
import logging
import os
import random

import tensorflow as tf

from model.utils import Params, set_logger, save_dict_to_json
import numpy as np
from model.input import imageload
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

    # Part 1: extract first digit data (Normal or abnormal)

    # define the paths to training, development, and test sets
    mypath_train = "./data/train"
    mypath_dev = "./data/dev"
    mypath_test = "./data/test"

    train_labels, train_data = imageload(mypath_train)
    dev_labels, dev_data = imageload(mypath_dev)
    test_labels, test_data = imageload(mypath_test)

    # numparize the array
    train_label_np, train_data_np = numparize(train_labels, train_data)
    dev_label_np, dev_data_np = numparize(dev_labels, dev_data)
    test_label_np, test_data_np = numparize(test_labels, test_data)

    # Describe the data
    height, width, channel, train_size, dev_size, test_size, categories, category_num = describe(
        train_label_np, train_data_np, dev_label_np, dev_data_np, test_label_np,
        test_data_np)

    # connecting to Keras below

    # Data extraction Part 2: extract first digit data (Left main or LAD or LCx or RCA)
    train_labels1, train_data1 = imageload(
        mypath_train, filename_label_position=1, dimheight=64,  dimwidth=64)
    dev_labels1, dev_data1 = imageload(
        mypath_dev, filename_label_position=1, dimheight=64,  dimwidth=64)
    test_labels1, test_data1 = imageload(
        mypath_test, filename_label_position=1, dimheight=64,  dimwidth=64)

    # numparize the array

    train_label_np1, train_data_np1 = numparize(train_labels1, train_data1)
    dev_label_np1, dev_data_np1 = numparize(dev_labels1, dev_data1)
    test_label_np1, test_data_np1 = numparize(test_labels1, test_data1)

    # Describe the data
    height1, width1, channel1, train_size1, dev_size1, test_size1, categories1, category_num1 = describe(
        train_label_np1, train_data_np1, dev_label_np1, dev_data_np1,
        test_label_np1, test_data_np1)

    # flatten the image and ensure it can go into Keras properly
    # notice the name is different from train_data.
    train_images = train_data_np.reshape((train_size, height, width, channel))
    train_images = train_images.astype('float32') / 255
    dev_images = dev_data_np.reshape((dev_size, height, width, channel))
    dev_images = dev_images.astype('float32') / 255
    test_images = test_data_np.reshape((test_size, height, width, channel))
    test_images = test_images.astype('float32') / 255

    train_images1 = train_data_np1.reshape((train_size1, height1, width1,
                                            channel1))
    train_images1 = train_images1.astype('float32') / 255
    dev_images1 = dev_data_np1.reshape((dev_size1, height1, width1, channel1))
    dev_images1 = dev_images1.astype('float32') / 255
    test_images1 = test_data_np1.reshape(
        (test_size1, height1, width1, channel1))
    test_images1 = test_images1.astype('float32') / 255

    # general categoricla variable
    # notice I am add "s" at train_label, to distinguish from earlier numpy.
    train_labels = to_categorical(train_label_np)
    dev_labels = to_categorical(dev_label_np)
    test_labels = to_categorical(test_label_np)

    train_labels1 = to_categorical(train_label_np1)
    dev_labels1 = to_categorical(dev_label_np1)
    test_labels1 = to_categorical(test_label_np1)

    # joining the two numpy arrray of one-shot into one
    # describe the train-label and label-1
    train_label_shape = np.shape(train_labels)
    train_label1_shape = np.shape(train_labels1)
    # so how many columns do we need in tatpe
    oneshot_column_merged = (train_label_shape[1] - 1) + train_label1_shape[1]
    sample_size = train_label_shape[0]  # sample size
    # create new numpy array of 0
    train_labels_merged = np.zeros((sample_size, oneshot_column_merged))

    train_labels_merged[0:sample_size, 0:(train_label_shape[1] - 1)] = train_labels[0:sample_size, 1].reshape(
        (sample_size, 1))  # put the smaller array into the bigger array
    train_labels_merged[0:sample_size,
                        (train_label_shape[1] - 1):oneshot_column_merged] = train_labels1

    # joining the two numpy arrray of one-shot into one
    # describe the train-label and label-1
    dev_label_shape = np.shape(dev_labels)
    dev_label1_shape = np.shape(dev_labels1)
    # so how many columns do we need in tatpe
    oneshot_column_merged = (dev_label_shape[1] - 1) + dev_label1_shape[1]
    sample_size = dev_label_shape[0]  # sample size
    # create new numpy array of 0
    dev_labels_merged = np.zeros((sample_size, oneshot_column_merged))

    dev_labels_merged[0:sample_size, 0:(dev_label_shape[1] - 1)] = dev_labels[0:sample_size, 1].reshape(
        (sample_size, 1))  # put the smaller array into the bigger array
    dev_labels_merged[0:sample_size,
                      (dev_label_shape[1] - 1):oneshot_column_merged] = dev_labels1

    # joining the two numpy arrray of one-shot into one
    # describe the train-label and label-1
    test_label_shape = np.shape(test_labels)
    test_label1_shape = np.shape(test_labels1)
    # so how many columns do we need in tatpe
    oneshot_column_merged = (test_label_shape[1] - 1) + test_label1_shape[1]
    sample_size = test_label_shape[0]  # sample size
    # create new numpy array of 0
    test_labels_merged = np.zeros((sample_size, oneshot_column_merged))

    test_labels_merged[0:sample_size, 0:(test_label_shape[1] - 1)] = test_labels[0:sample_size, 1].reshape(
        (sample_size, 1))  # put the smaller array into the bigger array
    test_labels_merged[0:sample_size,
                       (test_label_shape[1] - 1):oneshot_column_merged] = test_labels1
    for i in range(sample_size):
        print(test_labels_merged[i, :], ",",
              test_label_np[i], ",", test_label_np1[i])

    params = {}
    params['height'] = height
    params['width'] = width
    params['channel'] = channel

    model = build_model(True, params)
    history = train_model(model, train_labels_merged,
                          train_images, dev_labels_merged, dev_images)

    # print the graph of learning history for diagnostic purpose.
    print_plot_keras_metrics(history)
