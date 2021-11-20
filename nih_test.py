from tensorflow.keras.metrics import BinaryAccuracy
import argparse

from nih.data_generator import read_csv, ClassifyGenerator, l_diseases
from nih.model import create_model, FocalLoss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--csv_file', type=str, default='data/nih/train_val.csv')
    parser.add_argument('--image_dir', type=str, default='data/nih/images')
    parser.add_argument('--ckpt', type=str, default='ckpt/nih/checkpoint')
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    IMAGE_DIR = args['image_dir']
    BATCH_SIZE = args['batch_size']
    CSV_FILE = args['csv_file']

    # dataset
    X_test_df, y_test_df = read_csv(CSV_FILE)
    X_test = X_test_df.values.reshape(-1)
    test_ds = ClassifyGenerator(X_test, y_test_df.values, IMAGE_DIR, training=False, batch_size=args['batch_size'])

    # model
    model = create_model(weights=args['basenet_ckpt'])
    model.compile(loss=FocalLoss())

    # Predict
    accuracy = [BinaryAccuracy() for _ in l_diseases]
    for images, y_true in test_ds:
        y_pred = model(images, training=False)
        for i in range(len(accuracy)):
            accuracy[i](y_true[:, i], y_pred[:, i])

    for i in range(len(accuracy)):
        print(f"- {l_diseases[i]}: {accuracy[i].result()}")
