import math
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import densenet

from nih.configs import IMAGE_SIZE, l_diseases


def load_basenet(input_shape, weights=None):
    base_net = densenet.DenseNet201(input_shape=input_shape, include_top=False, weights='imagenet')
    if weights is not None:
        base_net.load_weights(weights).expect_partial()
    return base_net


def SPPLayer():
    def _pool(x, ksize):
        _, h, w, c = x.shape
        win_h = math.ceil(h / ksize)
        win_w = math.ceil(w / ksize)
        stride_h = math.floor(h / ksize)
        stride_w = math.floor(w / ksize)
        x = tf.nn.max_pool2d(x,
                             padding='VALID',
                             ksize=(win_h, win_w),
                             strides=(stride_h, stride_w))
        return tf.reshape(x, shape=(-1, ksize * ksize * c))

    def spp_pool(x):
        lv6 = _pool(x, ksize=6)
        lv5 = _pool(x, ksize=5)
        lv4 = _pool(x, ksize=4)
        lv3 = _pool(x, ksize=3)
        lv2 = _pool(x, ksize=2)
        lv1 = _pool(x, ksize=1)
        return tf.concat([lv1, lv2, lv3, lv4, lv5], axis=-1)

    return layers.Lambda(spp_pool)


def create_nih_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), weights=None, l2_decay=2e-5):
    spp_layer = SPPLayer()
    base_net = load_basenet(input_shape=input_shape, weights=weights)
    features = layers.Conv2D(128, kernel_size=3, padding='SAME')(base_net.output)
    features = spp_layer(features)
    features = layers.Dropout(0.3)(features)
    outputs = layers.Dense(len(l_diseases), kernel_regularizer=regularizers.l2(l2_decay))(features)
    return Model(inputs=[base_net.input], outputs=outputs)


class FocalLoss(tf.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.):
        super(FocalLoss, self).__init__()
        self._alpha = alpha
        self._gamma = gamma

    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        loss = tf.reduce_sum(loss, axis=-1)
        return tf.reduce_mean(loss)
