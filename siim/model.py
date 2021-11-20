import math
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import densenet

from nih.configs import IMAGE_SIZE, l_diseases
from nih.model import create_nih_model
from siim.configs import LABELS


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
        lv5 = _pool(x, ksize=5)
        lv4 = _pool(x, ksize=4)
        lv3 = _pool(x, ksize=3)
        lv2 = _pool(x, ksize=2)
        lv1 = _pool(x, ksize=1)
        return tf.concat([lv1, lv2, lv3, lv4, lv5], axis=-1)

    return layers.Lambda(spp_pool)


def create_siim_model(ckpt=None, l2_decay=1e-5):
    spp_layer = SPPLayer()
    nih_model = create_nih_model()
    nih_model.load_weights(ckpt).expect_partial()
    conv_layers = [l.name for l in nih_model.layers if 'conv' in l.name]
    features = nih_model.get_layer(conv_layers[-1]).output
    features = spp_layer(features)
    features = layers.Dropout(0.25)(features)
    outputs = layers.Dense(len(LABELS), kernel_regularizer=regularizers.l2(l2_decay))(features)
    return Model(inputs=[nih_model.input], outputs=outputs)
