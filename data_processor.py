import os
import re
import itertools
import traceback
from multiprocessing import Pool

import cv2
import numpy as np

from shapely.geometry import Polygon


# %%

def get_image_paths(data_path):
    allowed_extensions = ['jpg', 'png', 'jpeg', 'JPG']
    files = [os.path.join(data_path, file) for file in os.listdir(data_path) if
             any(file.endswith(ext) for ext in allowed_extensions)]
    return files


def get_text_file_path(image_path):
    text_file_path = image_path.replace(os.path.basename(image_path).split('.')[1], 'txt')
    return text_file_path


# %%

def shrink_poly(poly, r):
    # TODO check this for bugs
    # fit a poly inside the origin poly
    # used for generating the score map

    # shrinkratio
    R = 0.3
    # find the longer pair
    if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
            np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
        # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
        # p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        # p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
        # p0, p3
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        # p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
    else:
        # p0, p3
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        # p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
        # p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        # p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
    return poly


def fit_line(p1, p2):
    # fit a line ax+by+c = 0
    if p1[0] == p1[1]:
        return [1., 0., -p1[0]]
    else:
        [k, b] = np.polyfit(p1, p2, deg=1)
        return [k, -1., b]


def point_dist_to_line(p1, p2, p3):
    # compute the distance from p3 to p1-p2
    d = np.linalg.norm(p2 - p1)

    # what to if p1 and p2 coincide ?
    # alt = np.linalg.norm(p1 - p3)
    alt = np.nan

    return alt if d == 0.0 else np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / d


def line_cross_point(FLAGS, line1, line2):
    # line1 0= ax+by+c, compute the cross point of line1 and line2
    if line1[0] != 0 and line1[0] == line2[0]:
        if not FLAGS.suppress_warnings_and_error_messages:
            print('Cross point does not exist')
        return None
    if line1[0] == 0 and line2[0] == 0:
        if not FLAGS.suppress_warnings_and_error_messages:
            print('Cross point does not exist')
        return None
    if line1[1] == 0:
        x = -line1[2]
        y = line2[0] * x + line2[2]
    elif line2[1] == 0:
        x = -line2[2]
        y = line1[0] * x + line1[2]
    else:
        k1, _, b1 = line1
        k2, _, b2 = line2
        x = -(b1 - b2) / (k1 - k2)
        y = k1 * x + b1
    return np.array([x, y], dtype=np.float32)


def line_verticle(line, point):
    # get the verticle line from line across point
    if line[1] == 0:
        verticle = [0, -1, point[1]]
    else:
        if line[0] == 0:
            verticle = [1, 0, -point[0]]
        else:
            verticle = [-1. / line[0], -1, point[1] - (-1 / line[0] * point[0])]
    return verticle


def rectangle_from_parallelogram(FLAGS, poly):
    # fit a rectangle from a parallelogram
    p0, p1, p2, p3 = poly
    angle_p0 = np.arccos(np.dot(p1 - p0, p3 - p0) / (np.linalg.norm(p0 - p1) * np.linalg.norm(p3 - p0)))
    if angle_p0 < 0.5 * np.pi:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
            # p0 and p2
            ## p0
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p0)

            new_p3 = line_cross_point(FLAGS, p2p3, p2p3_verticle)
            ## p2
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p2)

            new_p1 = line_cross_point(FLAGS, p0p1, p0p1_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
        else:
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p0)

            new_p1 = line_cross_point(FLAGS, p1p2, p1p2_verticle)
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p2)

            new_p3 = line_cross_point(FLAGS, p0p3, p0p3_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
    else:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
            # p1 and p3
            ## p1
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p1)

            new_p2 = line_cross_point(FLAGS, p2p3, p2p3_verticle)
            ## p3
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p3)

            new_p0 = line_cross_point(FLAGS, p0p1, p0p1_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)
        else:
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p1)

            new_p0 = line_cross_point(FLAGS, p0p3, p0p3_verticle)
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p3)

            new_p2 = line_cross_point(FLAGS, p1p2, p1p2_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)


def sort_rectangle(FLAGS, poly):
    # sort the four coordinates of the polygon, points in poly should be sorted clockwise
    # first find the lowest point
    p_lowest = np.argmax(poly[:, 1])
    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        # if the bottom line is parallel to x-axis, then p0 must be the upper-left corner
        p0_index = np.argmin(np.sum(poly, axis=1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
    else:
        # find the point that sits right to the lowest point
        p_lowest_right = (p_lowest - 1) % 4
        p_lowest_left = (p_lowest + 1) % 4
        angle = np.arctan(
            -(poly[p_lowest][1] - poly[p_lowest_right][1]) / (poly[p_lowest][0] - poly[p_lowest_right][0]))
        # assert angle > 0
        if angle <= 0:
            if not FLAGS.suppress_warnings_and_error_messages:
                print(angle, poly[p_lowest], poly[p_lowest_right])
        if angle / np.pi * 180 > 45:
            # 这个点为p2 - this point is p2
            p2_index = p_lowest
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index + 1) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi / 2 - angle)
        else:
            # 这个点为p3 - this point is p3
            p3_index = p_lowest
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], angle


def generate_rbox(FLAGS, im_size, polys, tags):
    h, w = im_size
    shrinked_poly_mask = np.zeros((h, w), dtype=np.uint8)
    orig_poly_mask = np.zeros((h, w), dtype=np.uint8)
    score_map = np.zeros((h, w), dtype=np.uint8)
    geo_map = np.zeros((h, w, 5), dtype=np.float32)
    # mask used during traning, to ignore some hard areas
    overly_small_text_region_training_mask = np.ones((h, w), dtype=np.uint8)
    for poly_idx, poly_data in enumerate(zip(polys, tags)):
        poly = poly_data[0]
        tag = poly_data[1]

        r = [None, None, None, None]
        for i in range(4):
            r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                       np.linalg.norm(poly[i] - poly[(i - 1) % 4]))
        # score map
        shrinked_poly = shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]
        cv2.fillPoly(score_map, shrinked_poly, 1)
        cv2.fillPoly(shrinked_poly_mask, shrinked_poly, poly_idx + 1)
        cv2.fillPoly(orig_poly_mask, poly.astype(np.int32)[np.newaxis, :, :], 1)
        # if the poly is too small, then ignore it during training
        poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
        poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))
        if min(poly_h, poly_w) < FLAGS.min_text_size:
            cv2.fillPoly(overly_small_text_region_training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
        if tag:
            cv2.fillPoly(overly_small_text_region_training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

        xy_in_poly = np.argwhere(shrinked_poly_mask == (poly_idx + 1))
        # if geometry == 'RBOX':
        # generate a parallelogram for any combination of two vertices
        fitted_parallelograms = []
        for i in range(4):
            p0 = poly[i]
            p1 = poly[(i + 1) % 4]
            p2 = poly[(i + 2) % 4]
            p3 = poly[(i + 3) % 4]
            edge = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            backward_edge = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            forward_edge = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            if point_dist_to_line(p0, p1, p2) > point_dist_to_line(p0, p1, p3):
                # parallel lines through p2
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p2[0]]
                else:
                    edge_opposite = [edge[0], -1, p2[1] - edge[0] * p2[0]]
            else:
                # after p3
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p3[0]]
                else:
                    edge_opposite = [edge[0], -1, p3[1] - edge[0] * p3[0]]
            # move forward edge
            new_p0 = p0
            new_p1 = p1
            new_p2 = p2
            new_p3 = p3
            new_p2 = line_cross_point(FLAGS, forward_edge, edge_opposite)
            if point_dist_to_line(p1, new_p2, p0) > point_dist_to_line(p1, new_p2, p3):
                # across p0
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p0[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p0[1] - forward_edge[0] * p0[0]]
            else:
                # across p3
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p3[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p3[1] - forward_edge[0] * p3[0]]
            new_p0 = line_cross_point(FLAGS, forward_opposite, edge)
            new_p3 = line_cross_point(FLAGS, forward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
            # or move backward edge
            new_p0 = p0
            new_p1 = p1
            new_p2 = p2
            new_p3 = p3
            new_p3 = line_cross_point(FLAGS, backward_edge, edge_opposite)
            if point_dist_to_line(p0, p3, p1) > point_dist_to_line(p0, p3, p2):
                # across p1
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p1[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p1[1] - backward_edge[0] * p1[0]]
            else:
                # across p2
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p2[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p2[1] - backward_edge[0] * p2[0]]
            new_p1 = line_cross_point(FLAGS, backward_opposite, edge)
            new_p2 = line_cross_point(FLAGS, backward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
        areas = [Polygon(t).area for t in fitted_parallelograms]
        parallelogram = np.array(fitted_parallelograms[np.argmin(areas)][:-1], dtype=np.float32)
        # sort thie polygon
        parallelogram_coord_sum = np.sum(parallelogram, axis=1)
        min_coord_idx = np.argmin(parallelogram_coord_sum)
        parallelogram = parallelogram[
            [min_coord_idx, (min_coord_idx + 1) % 4, (min_coord_idx + 2) % 4, (min_coord_idx + 3) % 4]]

        rectange = rectangle_from_parallelogram(FLAGS, parallelogram)
        rectange, rotate_angle = sort_rectangle(FLAGS, rectange)

        p0_rect, p1_rect, p2_rect, p3_rect = rectange
        for y, x in xy_in_poly:
            point = np.array([x, y], dtype=np.float32)
            # top
            geo_map[y, x, 0] = point_dist_to_line(p0_rect, p1_rect, point)
            # right
            geo_map[y, x, 1] = point_dist_to_line(p1_rect, p2_rect, point)
            # down
            geo_map[y, x, 2] = point_dist_to_line(p2_rect, p3_rect, point)
            # left
            geo_map[y, x, 3] = point_dist_to_line(p3_rect, p0_rect, point)
            # angle
            geo_map[y, x, 4] = rotate_angle

    shrinked_poly_mask = (shrinked_poly_mask > 0).astype('uint8')
    text_region_boundary_training_mask = 1 - (orig_poly_mask - shrinked_poly_mask)

    return score_map, geo_map, overly_small_text_region_training_mask, text_region_boundary_training_mask


# %%

def load_annotation(txt_path):
    if not os.path.exists(txt_path):
        return np.array([], dtype=np.float32), np.array([], dtype=np.bool)

    file = open(txt_path)
    lines = file.readlines()
    file.close()

    def parse(line):
        line = line.split(',')
        x1, y1, x2, y2, x3, y3, x4, y4 = list(map(lambda item: float(re.sub(r'[^0-9.]+', '', item)), line[:8]))
        a = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        text = line[8].strip()
        b = text == '*' or text == '###'
        return a, b

    parsed = list(map(parse, lines))
    text_polys = [i[0] for i in parsed]
    text_tags = [i[1] for i in parsed]
    return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)


def polygon_area(poly):
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge) / 2.


def check_and_validate_polys(FLAGS, polys, tags, size):
    h, w = size
    if polys.shape[0] == 0:
        return polys
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)

    validated_polys = []
    validated_tags = []
    for poly, tag in zip(polys, tags):
        p_area = polygon_area(poly)
        if abs(p_area) < 1:
            if not FLAGS.suppress_warnings_and_error_messages:
                print('Invalid polygon.')
            continue
        if p_area > 0:
            if not FLAGS.suppress_warnings_and_error_messages:
                print('Polygon in wrong direction.')
            poly = poly[(0, 3, 2, 1), :]
        validated_polys.append(poly)
        validated_tags.append(tag)
    return np.array(validated_polys), np.array(validated_tags)


def pad_image(img, input_size, is_train):
    h, w, _ = img.shape
    max_h_w_i = np.max([h, w, input_size])
    img_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
    if is_train:
        shift_h = np.random.randint(max_h_w_i - h + 1)
        shift_w = np.random.randint(max_h_w_i - w + 1)
    else:
        shift_h = (max_h_w_i - h) // 2
        shift_w = (max_h_w_i - w) // 2
    img_padded[shift_h:h + shift_h, shift_w:w + shift_w, :] = img.copy()
    img = img_padded
    return img, shift_h, shift_w


def resize_image(img, text_polys, input_size, shift_h, shift_w):
    h, w, _ = img.shape
    img = cv2.resize(img, dsize=(input_size, input_size))
    resize_ratio_3_x = input_size / float(w)
    resize_ratio_3_y = input_size / float(h)
    text_polys[:, :, 0] += shift_w
    text_polys[:, :, 1] += shift_h
    text_polys[:, :, 0] *= resize_ratio_3_x
    text_polys[:, :, 1] *= resize_ratio_3_y
    return img, text_polys


# %%

def restore_rectangle_rbox(origin, geometry):
    d = geometry[:, :4]
    angle = geometry[:, 4]
    # for angle > 0
    origin_0 = origin[angle >= 0]
    d_0 = d[angle >= 0]
    angle_0 = angle[angle >= 0]
    if origin_0.shape[0] > 0:
        p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]),
                      np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]),
                      d_0[:, 3], -d_0[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_0 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_0 = np.zeros((0, 4, 2))
    # for angle < 0
    origin_1 = origin[angle < 0]
    d_1 = d[angle < 0]
    angle_1 = angle[angle < 0]
    if origin_1.shape[0] > 0:
        p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
                      -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
                      -d_1[:, 1], -d_1[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_1 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_1 = np.zeros((0, 4, 2))
    return np.concatenate([new_p_0, new_p_1])


def restore_rectangle(origin, geometry):
    return restore_rectangle_rbox(origin, geometry)


# %%

def crop_area(FLAGS, im, polys, tags, crop_background=False, max_tries=50):
    # make random crop from the input image
    h, w, _ = im.shape
    pad_h = h // 10
    pad_w = w // 10
    h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
    w_array = np.zeros((w + pad_w * 2), dtype=np.int32)
    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx + pad_w:maxx + pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny + pad_h:maxy + pad_h] = 1
    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        return im, polys, tags
    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w - 1)
        xmax = np.clip(xmax, 0, w - 1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h - 1)
        ymax = np.clip(ymax, 0, h - 1)
        if xmax - xmin < FLAGS.min_crop_side_ratio * w or ymax - ymin < FLAGS.min_crop_side_ratio * h:
            # area too small
            continue
        if polys.shape[0] != 0:
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
        else:
            selected_polys = []
        if len(selected_polys) == 0:
            # no text in this area
            if crop_background:
                return im[ymin:ymax + 1, xmin:xmax + 1, :], polys[selected_polys], tags[selected_polys]
            else:
                continue
        im = im[ymin:ymax + 1, xmin:xmax + 1, :]
        polys = polys[selected_polys]
        tags = tags[selected_polys]
        polys[:, :, 0] -= xmin
        polys[:, :, 1] -= ymin
        return im, polys, tags

    return im, polys, tags


# %%

def load_val_data_util(args):
    FLAGS, image_path = args
    try:
        img = cv2.imread(image_path)
        h, w, _ = img.shape
        txt_path = get_text_file_path(image_path)
        if not os.path.exists(txt_path):
            print('Text file {} does not exists'.format(txt_path))

        text_polys, text_tags = load_annotation(txt_path)
        text_polys, text_tags = check_and_validate_polys(FLAGS, text_polys, text_tags, (h, w))
        img, shift_h, shift_w = pad_image(img, FLAGS.input_size, is_train=False)
        img, text_polys = resize_image(img, text_polys, FLAGS.input_size, shift_h, shift_w)
        new_h, new_w, _ = img.shape

        score_map, geo_map, overly_small_text_region_training_mask, text_region_boundary_training_mask = generate_rbox(
            FLAGS, (new_h, new_w), text_polys, text_tags)

        img = (img / 127.5) - 1.

        return img[:, :, ::-1].astype(np.float32), image_path, score_map[::4, ::4, np.newaxis].astype(
            np.float32), geo_map[::4, ::4, :].astype(np.float32), overly_small_text_region_training_mask[::4, ::4,
                                                                  np.newaxis].astype(
            np.float32), text_region_boundary_training_mask[::4, ::4, np.newaxis].astype(np.float32)
    except Exception:
        if not FLAGS.suppress_warnings_and_error_messages:
            traceback.print_exc()


def load_val_data(FLAGS):
    image_files = np.array(get_image_paths(FLAGS.validation_data_path))

    pool = Pool(FLAGS.nb_workers)
    loaded_data = pool.map_async(load_val_data_util,
                                 zip(itertools.repeat(FLAGS), image_files)).get(
        timeout=9999999)

    pool.close()
    pool.join()

    images = [item[0] for item in loaded_data if item is not None]
    score_maps = [item[2] for item in loaded_data if item is not None]
    geo_maps = [item[3] for item in loaded_data if item is not None]
    overly_small_text_region_training_masks = [item[4] for item in loaded_data if item is not None]
    text_region_boundary_training_masks = [item[5] for item in loaded_data if item is not None]

    print('Number of validation images : %d' % len(images))

    return np.array(images), np.array(overly_small_text_region_training_masks), np.array(
        text_region_boundary_training_masks), np.array(score_maps), np.array(geo_maps)


# %%

def generator(FLAGS):
    image_paths = np.array(get_image_paths(FLAGS.training_data_path))
    print(f'{image_paths.shape[0]} training images in {FLAGS.training_data_path}')

    background_ratio = 3. / 8
    random_scale = np.array([0.5, 1, 2.0, 3.0])
    index = np.arange(0, image_paths.shape[0])

    while True:
        np.random.shuffle(index)
        images = []
        score_maps = []
        geo_maps = []
        overly_small_text_region_training_masks = []
        text_region_boundary_training_masks = []
        for i in index:
            try:
                image_path = image_paths[i]
                image = cv2.imread(image_path)
                h, w, _ = image.shape
                txt_path = get_text_file_path(image_path)
                if not os.path.exists(txt_path):
                    if not FLAGS.suppress_warnings_and_error_messages:
                        print(f'text file {txt_path} does not exists')
                    continue

                text_polys, text_tags = load_annotation(txt_path)
                text_polys, text_tags = check_and_validate_polys(FLAGS, text_polys, text_tags, (h, w))

                # random scale this image
                rd_scale = np.random.choice(random_scale)
                x_scale_variation = np.random.randint(-10, 10) / 100.
                y_scale_variation = np.random.randint(-10, 10) / 100.
                image = cv2.resize(image, dsize=None, fx=rd_scale + x_scale_variation, fy=rd_scale + y_scale_variation)
                text_polys[:, :, 0] *= rd_scale + x_scale_variation
                text_polys[:, :, 1] *= rd_scale + y_scale_variation

                # random crop a area from image
                crop_background = np.random.rand() < background_ratio
                image, text_polys, text_tags = crop_area(FLAGS, image, text_polys, text_tags,
                                                         crop_background=crop_background)
                if crop_background:
                    if text_polys.shape[0] > 0:
                        continue
                    image, _, _ = pad_image(image, FLAGS.input_size, is_train=True)
                    image = cv2.resize(image, dsize=(FLAGS.input_size, FLAGS.input_size))
                    score_map = np.zeros((FLAGS.input_size, FLAGS.input_size), dtype=np.uint8)
                    geo_map_channels = 5 if FLAGS.geometry == 'RBOX' else 8
                    geo_map = np.zeros((FLAGS.input_size, FLAGS.input_size, geo_map_channels), dtype=np.float32)
                    overly_small_text_region_training_mask = np.ones((FLAGS.input_size, FLAGS.input_size),
                                                                     dtype=np.uint8)
                    text_region_boundary_training_mask = np.ones((FLAGS.input_size, FLAGS.input_size), dtype=np.uint8)
                else:
                    if text_polys.shape[0] == 0:
                        continue
                    h, w, _ = image.shape
                    image, shift_h, shift_w = pad_image(image, FLAGS.input_size, is_train=True)
                    image, text_polys = resize_image(image, text_polys, FLAGS.input_size, shift_h, shift_w)
                    new_h, new_w, _ = image.shape
                    score_map, geo_map, overly_small_text_region_training_mask, text_region_boundary_training_mask = generate_rbox(
                        FLAGS, (new_h, new_w), text_polys, text_tags)

                image = (image / 127.5) - 1.
                images.append(image[:, :, ::-1].astype(np.float32))
                score_maps.append(score_map[::4, ::4, np.newaxis].astype(np.float32))
                geo_maps.append(geo_map[::4, ::4, :].astype(np.float32))
                overly_small_text_region_training_masks.append(
                    overly_small_text_region_training_mask[::4, ::4, np.newaxis].astype(np.float32))
                text_region_boundary_training_masks.append(
                    text_region_boundary_training_mask[::4, ::4, np.newaxis].astype(np.float32))

                if len(images) == FLAGS.batch_size:
                    yield [np.array(images), np.array(overly_small_text_region_training_masks),
                           np.array(text_region_boundary_training_masks), np.array(score_maps)], [np.array(score_maps),
                                                                                                  np.array(geo_maps)]
                    images = []
                    score_maps = []
                    geo_maps = []
                    overly_small_text_region_training_masks = []
                    text_region_boundary_training_masks = []

            except Exception:
                if not FLAGS.suppress_warnings_and_error_messages:
                    traceback.print_exc()
