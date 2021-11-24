import math
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import densenet

from common.spp_net import SPPLayer
from nih.configs import IMAGE_SIZE, l_diseases


def load_basenet(input_shape, weights=None):
    base_net = densenet.DenseNet201(input_shape=input_shape, include_top=False, weights='imagenet')
    if weights is not None:
        base_net.load_weights(weights).expect_partial()
    return base_net


def create_nih_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), weights=None, l2_decay=5e-5):
    spp_layer = SPPLayer()
    base_net = load_basenet(input_shape=input_shape, weights=weights)
    features = layers.Conv2D(128, kernel_size=3, padding='SAME', name='conv_out')(base_net.output)
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
