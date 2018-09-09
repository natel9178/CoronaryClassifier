#!/bin/bash
module load cudnn
python3 train.py --restore_from "experiments/weights/epochs_200_preprocess_testing_with_regularization_loss_2018-09-08_12-11-33_weights.chkpt.hdf5" --starting_epoch 139
