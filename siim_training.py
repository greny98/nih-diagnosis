from nih.model import FocalLoss
from siim.data_generator import SiimClassificationGenerator, split_data
from tensorflow.keras import optimizers, callbacks
import argparse

from siim.model import create_siim_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--csv_file', type=str, default='data/siim/train_val.csv')
    parser.add_argument('--image_dir', type=str, default='data/siim/images')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--output_dir', type=str, default='model')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--nih_ckpt', type=str, default=None)
    args = vars(parser.parse_args())
    return args


# Learning Rate Schedule
def schedule(e, lr):
    if e <= 3:
        return lr
    return 0.975 * lr


if __name__ == '__main__':
    args = parse_args()
    print(args)
    IMAGE_DIR = args['image_dir']
    BATCH_SIZE = args['batch_size']
    CSV_FILE = args['csv_file']

    # Create dataset
    train_data, val_data = split_data(CSV_FILE)
    train_ds = SiimClassificationGenerator(train_data, image_dir=IMAGE_DIR, batch_size=BATCH_SIZE,
                                           training=True)
    val_ds = SiimClassificationGenerator(val_data, image_dir=IMAGE_DIR, batch_size=BATCH_SIZE,
                                         training=False)
    # Config model
    model = create_siim_model(ckpt=args['nih_ckpt'])
    model.compile(
        loss=FocalLoss(),
        optimizer=optimizers.Adam(learning_rate=args['lr']))
    # Model checkpoints
    ckpt_cb = callbacks.ModelCheckpoint(
        filepath=f"{args['output_dir']}/checkpoint",
        save_best_only=True,
        save_weights_only=True,
        monitor='val_loss',
        mode='min'
    )

    lr_schedule_cb = callbacks.LearningRateScheduler(schedule)
    # Tensorboard
    tensorboard_cb = callbacks.TensorBoard(log_dir=args['log_dir'])
    # Fit
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args['epochs'],
        callbacks=[ckpt_cb, lr_schedule_cb, tensorboard_cb]
    )
