import tensorflow as tf
from tensorflow.keras import layers


class SPPLayer(layers.Layer):
    def __init__(self):
        super(SPPLayer, self).__init__()

    def _spp_pool(self, x, ksize):
        _, h, w, c = x.get_shape()
        win_h = tf.math.ceil(h / ksize)
        win_w = tf.math.ceil(w / ksize)
        stride_h = tf.math.floor(h / ksize)
        stride_w = tf.math.floor(w / ksize)
        x = tf.nn.max_pool2d(
            x,
            padding='VALID',
            ksize=(win_h, win_w),
            strides=(stride_h, stride_w))
        return tf.reshape(x, shape=(-1, ksize * ksize * c))

    def call(self, x, *args, **kwargs):
        lv4 = self._spp_pool(x, ksize=4)
        lv2 = self._spp_pool(x, ksize=2)
        lv1 = self._spp_pool(x, ksize=1)
        return tf.concat([lv1, lv2, lv4], axis=-1)
