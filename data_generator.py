import os
import traceback

import cv2
import numpy as np
from keras.utils import Sequence

from data_processor import get_image_paths, get_text_file_path, load_annotation, check_and_validate_polys, crop_area, \
    pad_image, resize_image, generate_rbox


class DataGenerator(Sequence):

    def __init__(self, input_size, batch_size, data_path, FLAGS, is_train=True):
        self.input_size = input_size
        self.batch_size = batch_size
        self.image_paths = get_image_paths(data_path)
        self.FLAGS = FLAGS
        self.is_train = is_train

    def __getitem__(self, index):

        images = []
        score_maps = []
        geo_maps = []
        overly_small_text_region_training_masks = []
        text_region_boundary_training_masks = []

        batch_image_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        for image_path in batch_image_paths:
            try:
                res = self.load_training(image_path) if self.is_train else self.load_validation(image_path)
                if res is not None:
                    image, score_map, geo_map, overly_small_text_region_training_mask, text_region_boundary_training_mask = res
                    images.append(image)
                    score_maps.append(score_map)
                    geo_maps.append(geo_map)
                    overly_small_text_region_training_masks.append(overly_small_text_region_training_mask)
                    text_region_boundary_training_masks.append(text_region_boundary_training_mask)
            except Exception:
                traceback.print_exc()

        return [np.array(images), np.array(overly_small_text_region_training_masks),
                np.array(text_region_boundary_training_masks), np.array(score_maps)], [np.array(score_maps),
                                                                                       np.array(geo_maps)]

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / float(self.batch_size)))

    def on_epoch_end(self):
        np.random.shuffle(self.image_paths)

    def load_training(self, image_path):
        FLAGS = self.FLAGS

        image = cv2.imread(image_path)
        h, w, _ = image.shape

        txt_path = get_text_file_path(image_path)
        if not os.path.exists(txt_path):
            return

        text_polys, text_tags = load_annotation(txt_path)
        text_polys, text_tags = check_and_validate_polys(FLAGS, text_polys, text_tags, (h, w))

        # random scale this image
        random_scale = np.array([0.5, 1, 2.0, 3.0])
        rd_scale = np.random.choice(random_scale)
        x_scale_variation = np.random.randint(-10, 10) / 100.
        y_scale_variation = np.random.randint(-10, 10) / 100.
        image = cv2.resize(image, dsize=None, fx=rd_scale + x_scale_variation, fy=rd_scale + y_scale_variation)
        text_polys[:, :, 0] *= rd_scale + x_scale_variation
        text_polys[:, :, 1] *= rd_scale + y_scale_variation

        # random crop a area from image
        background_ratio = 3. / 8
        crop_background = np.random.rand() < background_ratio
        image, text_polys, text_tags = crop_area(FLAGS, image, text_polys, text_tags,
                                                 crop_background=crop_background)
        if crop_background:
            if text_polys.shape[0] > 0:
                return
            image, _, _ = pad_image(image, self.input_size, is_train=True)
            image = cv2.resize(image, dsize=(self.input_size, self.input_size))
            score_map = np.zeros((self.input_size, self.input_size), dtype=np.uint8)
            geo_map_channels = 5 if FLAGS.geometry == 'RBOX' else 8
            geo_map = np.zeros((self.input_size, self.input_size, geo_map_channels), dtype=np.float32)
            overly_small_text_region_training_mask = np.ones((self.input_size, self.input_size),
                                                             dtype=np.uint8)
            text_region_boundary_training_mask = np.ones((self.input_size, self.input_size), dtype=np.uint8)
        else:
            if text_polys.shape[0] == 0:
                return
            h, w, _ = image.shape
            image, shift_h, shift_w = pad_image(image, self.input_size, is_train=True)
            image, text_polys = resize_image(image, text_polys, self.input_size, shift_h, shift_w)
            new_h, new_w, _ = image.shape
            score_map, geo_map, overly_small_text_region_training_mask, text_region_boundary_training_mask = generate_rbox(
                FLAGS, (new_h, new_w), text_polys, text_tags)

        image = (image / 127.5) - 1.
        return (
            image[:, :, ::-1].astype(np.float32),
            score_map[::4, ::4, np.newaxis].astype(np.float32),
            geo_map[::4, ::4, :].astype(np.float32),
            overly_small_text_region_training_mask[::4, ::4, np.newaxis].astype(np.float32),
            text_region_boundary_training_mask[::4, ::4, np.newaxis].astype(np.float32),
        )

    def load_validation(self, image_path):
        FLAGS = self.FLAGS

        image = cv2.imread(image_path)
        h, w, _ = image.shape

        txt_path = get_text_file_path(image_path)
        if not os.path.exists(txt_path):
            return

        text_polys, text_tags = load_annotation(txt_path)
        text_polys, text_tags = check_and_validate_polys(FLAGS, text_polys, text_tags, (h, w))
        image, shift_h, shift_w = pad_image(image, self.input_size, is_train=False)
        image, text_polys = resize_image(image, text_polys, self.input_size, shift_h, shift_w)
        new_h, new_w, _ = image.shape

        score_map, geo_map, overly_small_text_region_training_mask, text_region_boundary_training_mask = generate_rbox(
            FLAGS, (new_h, new_w), text_polys, text_tags)

        image = (image / 127.5) - 1.
        return (
            image[:, :, ::-1].astype(np.float32),
            score_map[::4, ::4, np.newaxis].astype(np.float32),
            geo_map[::4, ::4, :].astype(np.float32),
            overly_small_text_region_training_mask[::4, ::4, np.newaxis].astype(np.float32),
            text_region_boundary_training_mask[::4, ::4, np.newaxis].astype(np.float32)
        )
