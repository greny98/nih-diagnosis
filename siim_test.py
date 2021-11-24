from tensorflow.keras.metrics import BinaryAccuracy

from siim.configs import LABELS
from siim.data_generator import split_data, SiimClassificationGenerator
from siim.model import create_siim_model
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--csv_file', type=str, default='data/siim/train_val.csv')
    parser.add_argument('--image_dir', type=str, default='data/siim/images')
    parser.add_argument('--ckpt', type=str, default='ckpt/classification1/checkpoint')
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    IMAGE_DIR = args['image_dir']
    BATCH_SIZE = args['batch_size']
    CSV_FILE = args['csv_file']

    # dataset
    train_data, val_data = split_data(CSV_FILE)
    val_ds = SiimClassificationGenerator(val_data, image_dir=IMAGE_DIR, batch_size=BATCH_SIZE,
                                         training=False)
    accuracy = [BinaryAccuracy() for _ in LABELS]
    # load model
    model = create_siim_model(ckpt=args['nih_ckpt'])
    model.load_weights(args['ckpt'])

    # Predict
    for images, y_true in val_ds:
        y_pred = model(images, training=False)
        for i in range(len(accuracy)):
            accuracy[i](y_true[:, i], y_pred[:, i])

    for i in range(len(accuracy)):
        print(f"- {LABELS[i]}: {accuracy[i].result()}")
