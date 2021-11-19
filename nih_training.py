from nih.data_generator import ClassifyGenerator, read_csv, train_val_split
from nih.model import create_model, FocalLoss
from tensorflow.keras import optimizers, callbacks
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--csv_file', type=str, default='data/nih/train_val.csv')
    parser.add_argument('--image_dir', type=str, default='data/nih/images/')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--output_dir', type=str, default='model')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--basenet_ckpt', type=str, default=None)
    args = vars(parser.parse_args())
    return args


# Learning Rate Schedule
def schedule(e, lr):
    if e <= 5 or e % 3 != 0:
        return lr
    return 0.95 * lr


if __name__ == '__main__':
    args = parse_args()
    print(args)
    IMAGE_DIR = args['image_dir']
    BATCH_SIZE = args['batch_size']
    CSV_FILE = args['csv_file']

    # Create dataset
    X_train_val_df, y_train_val_df = read_csv(CSV_FILE)
    # Split train, val
    (X_train, y_train), (X_val, y_val) = train_val_split(X_train_val_df.values, y_train_val_df.values, log=False)
    # Flatten X
    X_train = X_train.reshape(-1)
    X_val = X_val.reshape(-1)
    # Create ds
    train_ds = ClassifyGenerator(X_train, y_train, IMAGE_DIR, training=True, batch_size=args['batch_size'])
    val_ds = ClassifyGenerator(X_val, y_val, IMAGE_DIR, training=False, batch_size=args['batch_size'])

    # Config model
    model = create_model(weights=args['basenet_ckpt'])
    model.compile(loss=FocalLoss(), optimizer=optimizers.Adam(learning_rate=args['lr']))
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
