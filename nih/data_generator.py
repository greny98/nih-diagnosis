import numpy as np
import pandas as pd
import tensorflow as tf
import albumentations as augment
from skmultilearn.model_selection import iterative_train_test_split
from nih.configs import l_diseases, IMAGE_SIZE
from tensorflow.keras.applications import densenet

autotune = tf.data.AUTOTUNE


def split_labels(df: pd.DataFrame):
    X_col = df[['Image Index']]
    y_cols = df[l_diseases]
    return X_col, y_cols


def read_csv(filename: str):
    df = pd.read_csv(filename)
    return split_labels(df)


def train_val_split(X_train_val, y_train_val, test_size=0.1, log=False):
    X_train, y_train, X_val, y_val = iterative_train_test_split(X_train_val, y_train_val, test_size=test_size)
    if log:
        print("Split information:")
        pos_ratio = y_train.sum(axis=0) / y_val.sum(axis=0)
        pos_train_ratio = np.mean(y_train, axis=0)
        pos_val_ratio = np.mean(y_val, axis=0)
        print("- Ratio:")
        for i, disease in enumerate(l_diseases):
            print(
                f"\t+ {disease}: train/val={pos_ratio[i]} - pos_train: {pos_train_ratio[i]} - pos_val: {pos_val_ratio[i]}")
    return (X_train, y_train), (X_val, y_val)


def classify_augmentation(training=False):
    if training:
        transform = augment.Compose([
            augment.ImageCompression(quality_lower=90, quality_upper=100, p=0.4),
            augment.SmallestMaxSize(720),
            augment.HorizontalFlip(),
            augment.VerticalFlip(),
            augment.RandomRotate90(),
            augment.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            augment.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.4),
            augment.GaussNoise(p=0.4),
            augment.GaussianBlur(p=0.4),
            augment.RandomSizedCrop(min_max_height=(512, 720), height=IMAGE_SIZE, width=IMAGE_SIZE, w2h_ratio=1.)
        ])
    else:
        transform = augment.Compose([augment.Resize(IMAGE_SIZE, IMAGE_SIZE)])

    def preprocess_image(image_file):
        image_raw = tf.io.read_file(image_file)
        decoded = tf.image.decode_jpeg(image_raw, channels=3)
        data = {'image': decoded.numpy()}
        aug_img = transform(**data)['image']
        aug_img = tf.cast(aug_img, tf.float32)
        tensor = densenet.preprocess_input(aug_img)
        return tensor

    return preprocess_image


def ClassifyGenerator(images, y, image_dir, batch_size, training=False):
    def process_data(image_file, label):
        aug_img = tf.numpy_function(func=classify_augmentation(training), inp=[image_file], Tout=tf.float32)
        return aug_img, label

    images_ts = tf.data.Dataset.from_tensor_slices(image_dir + images)
    labels_ts = tf.data.Dataset.from_tensor_slices(y.astype(float))
    ds = tf.data.Dataset.zip((images_ts, labels_ts))
    ds = ds.shuffle(128 * batch_size, reshuffle_each_iteration=training)
    ds = ds.map(lambda x, y: process_data(x, y),
                num_parallel_calls=autotune).batch(batch_size)
    ds = ds.prefetch(autotune)
    return ds
