
import argparse
import logging
import os
import random
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the combined dataset")


if __name__ == '__main__':
    args = parser.parse_args()

    filenames = os.listdir(args.data_dir)
    filenames.sort()  # make sure that the filenames have a fixed order before shuffling
    random.seed(230)
    # shuffles the ordering of filenames (deterministic given the chosen seed)
    random.shuffle(filenames)

    split_1 = int(0.8 * len(filenames))
    split_2 = int(0.9 * len(filenames))

    train_filenames = filenames[:split_1]
    dev_filenames = filenames[split_1:split_2]
    test_filenames = filenames[split_2:]

    if not os.path.exists(os.path.join(args.data_dir, 'train')):
        os.makedirs(os.path.join(args.data_dir, 'train'))
    if not os.path.exists(os.path.join(args.data_dir, 'dev')):
        os.makedirs(os.path.join(args.data_dir, 'dev'))
    if not os.path.exists(os.path.join(args.data_dir, 'test')):
        os.makedirs(os.path.join(args.data_dir, 'test'))

    for x in train_filenames:
        shutil.move(os.path.join(args.data_dir, x),
                    os.path.join(args.data_dir, 'train', x))

    for x in dev_filenames:
        shutil.move(os.path.join(args.data_dir, x),
                    os.path.join(args.data_dir, 'dev', x))

    for x in test_filenames:
        shutil.move(os.path.join(args.data_dir, x),
                    os.path.join(args.data_dir, 'test', x))
