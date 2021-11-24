import math
import tensorflow as tf
from tensorflow.keras import layers


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
        return tf.concat([lv1, lv2, lv3, lv4, lv5, lv6], axis=-1)

    return layers.Lambda(spp_pool)
