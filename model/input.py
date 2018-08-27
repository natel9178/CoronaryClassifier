import os
import cv2
import matplotlib
import numpy as np


def imageload(mypath_train, mypath_dev, mypath_test, filename_label_position,
              dimheight, dimwidth):
    train_label = {}  # creat train_label and train_data dictionary
    train_data = {}
    set_of_file = set()
    i = 0
    for root, dir, files in os.walk(
            mypath_train):  # walk through all files in the folder
        for file in files:
            if file[0] == ".":
                continue
            if file in set_of_file:
                continue
            set_of_file.add(file)  # make sure file not duplicate
            mypath_file = os.path.join(mypath_train,
                                       file)  # define the path with directory
            img = cv2.imread(mypath_file)  # read with OPenCV first
            print(mypath_file)

            # read with Open CV. note: expect color images
            resized_img = cv2.resize(
                img, (dimheight, dimwidth),
                interpolation=cv2.INTER_AREA)  # resize with OpenCV
            train_label[i] = file[
                filename_label_position]  # obtain label of data (the character position is used to indicate categories)

            img1 = np.array(
                resized_img)  # numparize the resized image into a numpay array
            train_data[
                i] = img1  #store the numpay array into a dictionary (here this is train_data)
            i += 1
    return train_label, train_data, dev_label, dev_data, test_label, test_data
