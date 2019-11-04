import numpy as np

import tensorflow as tf

from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, concatenate, Conv2D, BatchNormalization, Activation
from tensorflow.keras import regularizers


def get_activation_layers_from_resnet(resnet):
    activation_layers = {}
    idx = 1
    for layer in resnet.layers:
        if 'Activation' in str(type(layer)):
            activation_layers[f'activation_{idx}'] = layer
            idx += 1
    return activation_layers


RESIZE_FACTOR = 2


def resize_bilinear(x):
    return tf.compat.v1.image.resize_bilinear(x, size=[x.shape[1] * RESIZE_FACTOR, x.shape[2] * RESIZE_FACTOR])


def resize_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4
    shape[1] *= RESIZE_FACTOR
    shape[2] *= RESIZE_FACTOR
    return tuple(shape)


class EastModel:

    def __init__(self, input_size=512):
        input_image = Input(shape=(input_size, input_size, 3), name='input_image')
        overly_small_text_region_training_mask = Input(shape=(input_size // 4, input_size // 4, 1),
                                                       name='overly_small_text_region_training_mask')
        text_region_boundary_training_mask = Input(shape=(input_size // 4, input_size // 4, 1),
                                                   name='text_region_boundary_training_mask')
        target_score_map = Input(shape=(input_size // 4, input_size // 4, 1), name='target_score_map')

        resnet = ResNet50(input_tensor=input_image,
                          weights='imagenet', include_top=False, pooling=None)
        activation_layers = get_activation_layers_from_resnet(resnet)

        x = activation_layers['activation_49'].output
        x = Lambda(resize_bilinear, name='resize_1')(x)
        x = concatenate([x, activation_layers['activation_40'].output], axis=3)
        x = Conv2D(128, (1, 1), padding='same',
                   kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        x = Lambda(resize_bilinear, name='resize_2')(x)
        x = concatenate([x, activation_layers['activation_22'].output], axis=3)
        x = Conv2D(64, (1, 1), padding='same',
                   kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        x = Lambda(resize_bilinear, name='resize_3')(x)
        x = concatenate([x, activation_layers['activation_10'].output], axis=3)
        x = Conv2D(32, (1, 1), padding='same',
                   kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(32, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        x = Conv2D(32, (3, 3), padding='same',
                   kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        pred_score_map = Conv2D(
            1, (1, 1), activation=tf.nn.sigmoid, name='pred_score_map')(x)
        rbox_geo_map = Conv2D(
            4, (1, 1), activation=tf.nn.sigmoid, name='rbox_geo_map')(x)
        rbox_geo_map = Lambda(lambda x: x * input_size)(rbox_geo_map)
        angle_map = Conv2D(1, (1, 1), activation=tf.nn.sigmoid,
                           name='rbox_angle_map')(x)
        angle_map = Lambda(lambda x: (x - 0.5) * np.pi / 2)(angle_map)
        pred_geo_map = concatenate(
            [rbox_geo_map, angle_map], axis=3, name='pred_geo_map')

        model = Model(inputs=[input_image, overly_small_text_region_training_mask,
                              text_region_boundary_training_mask, target_score_map],
                      outputs=[pred_score_map, pred_geo_map])

        self.model = model
        self.input_image = input_image
        self.overly_small_text_region_training_mask = overly_small_text_region_training_mask
        self.text_region_boundary_training_mask = text_region_boundary_training_mask
        self.target_score_map = target_score_map
        self.pred_score_map = pred_score_map
        self.pred_geo_map = pred_geo_map


if __name__ == '__main__':
    EAST()
