
'''
Convert dictionaries train_label, dev_label, test-label,  (train_data, dev_data, test_data into numpy array
assume that all images are same shape
'''

import numpy as np


def numparize(labels, data):

    # 1. check if label and data set have same number
    if len(labels) != len(data):
        print("error in data and label size")
        exit()

    # 2 obtaining size of the dictionaries in numpy arrays
    len_labels = len(labels)  # all these sets were dictionary

    # train_data is supposed to be a dictionary of (index, numpy array of resized image).
    imageshape = np.shape(data[0])
    # we only look at the first one
    # the height and width were obtained from the shaoe abive
    height, width, channel = imageshape
    # expect color image

    # 3 Initializing new numpy array
    label_np = np.zeros((len_labels, 1))

    data_np = np.zeros((len_labels, height, width, channel))

    # 4  converting data to numpy

    for i in range(len_labels):
        data_np[i, :, :, :] = data[i]
        label_np[i] = labels[i]

    return label_np, data_np
