#!/bin/bash
module load cudnn
#python3 train.py --restore_from ./experiments/weights/epochs_200_datav2_densenet_regularize_0.01_2018-09-08_20-32-08_weights.chkpt.hdf5 --starting_epoch 179 --learning_rate 0.0001
python3 train.py
