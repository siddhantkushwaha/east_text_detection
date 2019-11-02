import os
import argparse

import cv2
import numpy as np

import tensorflow as tf
from keras.models import model_from_json

import lanms
from data_processor import get_image_paths, get_text_file_path, restore_rectangle
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


def resize_image(im, max_side_len=2400):
    # resize image to a size multiple of 32 which is required by the network
    # max_side_len: limit of max image size to avoid out of memory in gpu

    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    # restore text boxes from score map and geo map
    # param score_map:
    # param geo_map:
    # param score_map_thresh: threshhold for score map
    # param box_thresh: threshhold for boxes
    # param nms_thres: threshold for nms

    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]

    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)

    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]

    # restore
    text_box_restored = restore_rectangle(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]

    # nms part
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    if boxes.shape[0] == 0:
        return None

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]

    boxes = boxes[boxes[:, 8] > box_thresh]
    return boxes


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def main():
    os.system(f'mkdir -p {FLAGS.output_dir}')

    model = load_model()

    image_paths = get_image_paths(FLAGS.test_data_path)
    data = list(map(lambda image_path: (image_path, get_text_file_path(image_path)), image_paths))
    for image_path, txt_path in data:
        print(image_path)
        img = cv2.imread(image_path)
        img = img[:, :, ::-1]
        img_resized, (ratio_h, ratio_w) = resize_image(img)
        img_resized = (img_resized / 127.5) - 1

        score_map, geo_map = model.predict(img_resized[np.newaxis, :, :, :])

        boxes = detect(score_map=score_map, geo_map=geo_map)
        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

            res_file = os.path.join(FLAGS.output_dir, '{}.txt'.format(os.path.basename(image_path).split('.')[0]))
            with open(res_file, 'w') as f:
                for box in boxes:
                    box = sort_poly(box.astype(np.int32))
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                        continue
                    f.write('{},{},{},{},{},{},{},{}\r\n'.format(box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0],
                                                                 box[2, 1], box[3, 0], box[3, 1], ))
                    cv2.polylines(img[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True,
                                  color=(255, 255, 0), thickness=1)

            out_image_path = os.path.join(FLAGS.output_dir, os.path.basename(image_path))
            cv2.imwrite(out_image_path, img[:, :, ::-1])


if __name__ == '__main__':
    main()
