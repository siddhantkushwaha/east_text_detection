import os
import argparse

import tensorflow as tf
from keras.models import model_from_json

from model import RESIZE_FACTOR

parser = argparse.ArgumentParser()
parser.add_argument('--test_data_path', type=str, default='data/ICDAR2015/test_data/')
parser.add_argument('--model_path', type=str, default='models/east_v1/model.h5')
parser.add_argument('--output_dir', type=str, default='out/')
FLAGS = parser.parse_args()


def load_model():
    json_path = '/'.join(FLAGS.model_path.split('/')[0:-1])
    file = open(os.path.join(json_path, 'model.json'), 'r')
    model_json = file.read()
    file.close()

    model = model_from_json(model_json, custom_objects={'tf': tf, 'RESIZE_FACTOR': RESIZE_FACTOR})
    model.load_weights(FLAGS.model_path)
    return model


def main():
    model = load_model()
    print(model.summary())


if __name__ == '__main__':
    main()
