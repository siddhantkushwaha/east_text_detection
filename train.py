import os
import logging
import argparse
from datetime import datetime

import keras.backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint

from adamw import AdamW
from losses import dice_loss, rbox_loss
from model import EastModel

from data_generator import DataGenerator

parser = argparse.ArgumentParser()

parser.add_argument('--training_data_path', type=str, default='../ICDAR2015/train_data')
parser.add_argument('--validation_data_path', type=str, default='../ICDAR2015/val_data')
parser.add_argument('--checkpoint_path', type=str, default='models/east')

parser.add_argument('--input_size', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--nb_workers', type=int, default=16)
parser.add_argument('--max_epochs', type=int, default=150)
parser.add_argument('--init_learning_rate', type=float, default=0.0001)
parser.add_argument('--save_checkpoint_epochs', type=int, default=10)

parser.add_argument('--min_text_size', type=int, default=10)
parser.add_argument('--min_crop_side_ratio', type=float, default=0.1)
parser.add_argument('--geometry', type=str, default='RBOX')
parser.add_argument('--suppress_warnings_and_error_messages', type=bool, default=True)

FLAGS = parser.parse_args()


def tensorboard_callback():
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    return TensorBoard(log_dir=log_dir)


def checkpoint_callback():
    os.system(f'mkdir -p {FLAGS.checkpoint_path}')
    return ModelCheckpoint(
        filepath=os.path.join(FLAGS.checkpoint_path, 'model-{epoch:02d}-{val_loss:.2f}.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=0,
        save_weights_only=True,
        period=FLAGS.save_checkpoint_epochs
    )


def main():
    train_data_generator = DataGenerator(input_size=FLAGS.input_size, batch_size=FLAGS.batch_size,
                                         data_path=FLAGS.training_data_path, FLAGS=FLAGS, is_train=True)
    train_samples_count = len(train_data_generator.image_paths)
    validation_data_generator = DataGenerator(input_size=FLAGS.input_size, batch_size=FLAGS.batch_size,
                                              data_path=FLAGS.validation_data_path, FLAGS=FLAGS, is_train=False)

    east = EastModel(FLAGS.input_size)

    score_map_loss_weight = K.variable(0.01, name='score_map_loss_weight')
    small_text_weight = K.variable(0., name='small_text_weight')

    opt = AdamW(FLAGS.init_learning_rate)
    east.model.compile(
        loss=[
            dice_loss(east.overly_small_text_region_training_mask, east.text_region_boundary_training_mask,
                      score_map_loss_weight, small_text_weight),
            rbox_loss(east.overly_small_text_region_training_mask, east.text_region_boundary_training_mask,
                      small_text_weight, east.target_score_map)
        ],
        loss_weights=[1., 1.],
        optimizer=opt,
    )

    tb_callback = tensorboard_callback()
    cp_callback = checkpoint_callback()

    with open(os.path.join(FLAGS.checkpoint_path, 'model.json'), 'w') as json_file:
        json_file.write(east.model.to_json())

    east.model.fit_generator(
        generator=train_data_generator,
        epochs=FLAGS.max_epochs,
        steps_per_epoch=train_samples_count // FLAGS.batch_size,
        validation_data=validation_data_generator,

        callbacks=[cp_callback, tb_callback],

        workers=FLAGS.nb_workers,
        use_multiprocessing=True,
        max_queue_size=10,

        verbose=1,
    )


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    main()
