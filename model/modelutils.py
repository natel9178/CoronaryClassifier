
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


def describe(train_label_np, train_data_np, dev_label_np, dev_data_np):
    # this takes numpy array training data and return: height, width, train_size, dev_size, test_size,categories, category_num
    import numpy as np
    train_size, height, width, channel = np.shape(train_data_np)

    dev_size, _, _, _ = np.shape(dev_data_np)
    categories = np.unique(train_label_np)
    category_num = len(categories)

    print("Sanity Check:")
    print("height=", height, " width=", width, " channel=", channel)
    print("training set size=", train_size,
          "   development set size=", dev_size)
    print("categories=", categories)
    print("number of category=", category_num)

    print("------------")

    return height, width, channel, train_size, dev_size, categories, category_num
