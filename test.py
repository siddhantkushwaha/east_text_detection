import argparse
from data_processor import *

parser = argparse.ArgumentParser()

parser.add_argument('--training_data_path', type=str, default='data/ICDAR2015/train_data')
parser.add_argument('--validation_data_path', type=str, default='data/ICDAR2015/val_data')
parser.add_argument('--checkpoint_path', type=str, default='models/east_v1')

parser.add_argument('--input_size', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--nb_workers', type=int, default=4)
parser.add_argument('--init_learning_rate', type=float, default=0.0001)
parser.add_argument('--lr_decay_rate', type=float, default=0.94)
parser.add_argument('--lr_decay_steps', type=int, default=130)
parser.add_argument('--max_epochs', type=int, default=150)
parser.add_argument('--save_checkpoint_epochs', type=int, default=10)

parser.add_argument('--gpu_list', type=str, default='0')
parser.add_argument('--restore_model', type=str, default='')
parser.add_argument('--max_image_large_side', type=int, default=1280)
parser.add_argument('--max_text_size', type=int, default=800)
parser.add_argument('--min_text_size', type=int, default=10)
parser.add_argument('--min_crop_side_ratio', type=float, default=0.1)
parser.add_argument('--geometry', type=str, default='RBOX')
parser.add_argument('--suppress_warnings_and_error_messages', type=bool, default=True)

FLAGS = parser.parse_args()

if __name__ == '__main__':
    val_data = load_val_data(FLAGS)

    train_data = generator(FLAGS)
    for x in train_data:
        print(x[0][0].shape)
