import os.path
import time

from nih.configs import l_diseases
from nih.data_generator import ClassifyGenerator, read_csv, train_val_split
from nih.model import FocalLoss, DiagnosisModel
from tensorflow.keras import optimizers, metrics
import argparse
import tensorflow as tf


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
    if not os.path.exists(args['output_dir']):
        os.makedirs(args['output_dir'])
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
    train_ds = ClassifyGenerator(X_train, y_train, IMAGE_DIR, training=True)
    val_ds = ClassifyGenerator(X_val, y_val, IMAGE_DIR, training=False)

    # Config model
    model = DiagnosisModel(len(l_diseases))


    def training_steps(x, y_true):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            y_true = tf.cast(y_true, dtype=tf.float32)
            loss_val = loss_fn(y_true, y_pred)
        grads = tape.gradient(loss_val, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_val


    def validate_steps(x, y_true):
        y_pred = model(x, training=False)
        y_true = tf.cast(y_true, dtype=tf.float32)
        return loss_fn(y_true, y_pred)


    loss_fn = FocalLoss()
    training_loss_mean = metrics.Mean(name="loss")
    validate_loss_mean = metrics.Mean(name="val_loss")

    epochs = args["epochs"]
    lr = args['lr']
    decay_lr = 0.95
    best_val_loss = None
    print("Total Train: ", len(train_ds))
    print("Total Val: ", len(val_ds))
    for e in range(epochs):
        if e > 5 and e % 4 == 0:
            lr = lr * decay_lr
        optimizer = optimizers.Adam(learning_rate=lr)
        # reset metrics state
        training_loss_mean.reset_states()
        validate_loss_mean.reset_states()
        # training
        start = time.time()
        for step, (images, labels) in enumerate(train_ds):
            training_loss = training_steps(images, labels)
            training_loss_mean(training_loss)
            if (step+1) % 1000 == 0:
                print(f"Step {step+1}:", training_loss_mean.result().numpy())
        print("Train time:", time.time() - start)
        # validate
        start = time.time()
        for step, (images, labels) in enumerate(val_ds):
            val_loss = validate_steps(images, labels)
            validate_loss_mean(val_loss)
        print("Val time:", time.time() - start)
        if (best_val_loss is None) or (validate_loss_mean.result().numpy() < best_val_loss):
            model.save_weights(f"{args['output_dir']}/checkpoint")
        print(f"Epoch {e}:")
        print("\t- Training Loss =", training_loss_mean.result().numpy())
        print("\t- Validation Loss =", validate_loss_mean.result().numpy())
