import os
import cv2
import matplotlib
import numpy as np


def imageload(data_directory, filename_label_position=0, dimheight=128, dimwidth=128):
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
