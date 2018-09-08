import os
import cv2
import matplotlib
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

'''
Call to load images into initial arrays,
the data file digits preceding the "-" are the label, for instance filename_label_position=0,
means the position 0 (first digit) is the label of interest
in our dataset, first digit is normal (=0), vs abnormal (=1)
second digit is which artery (0=left main, 1=LAD, 2= LCx, 3= RCA)
the picture is converted to dimenston height/difeth set below, using CV2)
'''


def imageload(data_directory, filename_label_position=0, dimheight=224, dimwidth=224):
    labels = {}  # creat train_label and train_data dictionary
    data = {}
    filenames = set()
    i = 0

    # walk through all files in the folder
    for _, _, files in os.walk(data_directory):
        for file in files:
            if file[0] == "." or file in filenames:
                continue

            filenames.add(file)  # make sure file not duplicate
            # define the path with directory
            mypath_file = os.path.join(data_directory, file)
            img = cv2.imread(mypath_file)  # read with OPenCV first

            # read with Open CV. note: expect color images
            # resize with OpenCV
            resized_img = cv2.resize(
                img, (dimheight, dimwidth), interpolation=cv2.INTER_AREA)
            # obtain label of data (the character position is used to indicate categories)
            labels[i] = file[filename_label_position]

            # numparize the resized image into a numpay array
            img1 = np.array(resized_img)

            # store the numpay array into a dictionary (here this is data)
            data[i] = img1

            i += 1

    return labels, data


def expose_generators(train_data, train_labels_stenosis, train_labels_anatomy, val_data, val_labels_stenosis, val_labels_anatomy, batch_size=32):
    shift = 0.2
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        zoom_range=0.2, width_shift_range=shift, height_shift_range=shift)

    val_datagen = ImageDataGenerator(rescale=1./255)

    seed = 1
    train_datagen.fit(train_data, augment=True, seed=seed)
    val_datagen.fit(val_data, augment=True, seed=seed)

    print(train_labels_stenosis.shape)
    print(train_labels_anatomy.shape)
    train_y = [train_labels_stenosis, train_labels_anatomy]
    val_y = [val_labels_stenosis, val_labels_anatomy]
    train_flow_data = (train_data, train_y)
    val_flow_data = (val_data, val_y)

    train_flow = train_datagen.flow(train_flow_data,
                                    batch_size=batch_size, shuffle=True, seed=seed)
    val_flow = val_datagen.flow(val_flow_data, batch_size=batch_size,
                                shuffle=True, seed=seed)

    return train_flow, val_flow


def generator_hotfix(generator):
    while True:
        next = generator.next()
        yield next[0], [next[1], next[2]]


def merge_labels(binary_labels, categorical_labels):
    # joining the two numpy arrray of one-shot into one
    # describe the train-label and label-1
    binary_labels_shape = np.shape(binary_labels)
    categorical_labels_shape = np.shape(categorical_labels)
    # so how many columns do we need in tatpe
    oneshot_column_merged = (
        binary_labels_shape[1] - 1) + categorical_labels_shape[1]
    sample_size = binary_labels_shape[0]  # sample size
    # create new numpy array of 0
    labels_merged = np.zeros((sample_size, oneshot_column_merged))

    labels_merged[0:sample_size, 0:(binary_labels_shape[1] - 1)] = binary_labels[0:sample_size, 1].reshape(
        (sample_size, 1))  # put the smaller array into the bigger array
    labels_merged[0:sample_size,
                  (binary_labels_shape[1] - 1):oneshot_column_merged] = categorical_labels

    return labels_merged
