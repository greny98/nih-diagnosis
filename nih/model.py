import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers, losses
from tensorflow.keras.applications import densenet

from common.densenet import DenseNet
from common.spp_net import SPPLayer
from nih.configs import l_diseases


def load_basenet(input_shape, weights=None):
    base_net = densenet.DenseNet201(input_shape=input_shape, include_top=False, weights='imagenet')
    if weights is not None:
        base_net.load_weights(weights).expect_partial()
    return base_net


def create_nih_model(weights=None, l2_decay=2e-5):
    spp_layer = SPPLayer()
    base_net = load_basenet(input_shape=(None, None, 3), weights=weights)
    features = layers.Conv2D(128, kernel_size=3, padding='SAME', name='conv_out')(base_net.output)
    features = spp_layer(features)
    features = layers.Dropout(0.25)(features)
    outputs = layers.Dense(len(l_diseases), kernel_regularizer=regularizers.l2(l2_decay))(features)
    return Model(inputs=[base_net.input], outputs=outputs)


class SPPNetModel(Model):
    def __init__(self, basenet_weights=None):
        super(SPPNetModel, self).__init__(name="SPPNet")
        self.basenet = DenseNet()
        if basenet_weights is not None:
            self.basenet.load_weights(basenet_weights)
        self.spp_pool = SPPLayer()
        self.conv_out = layers.Conv2D(128, kernel_size=3, padding='SAME', name='conv_out')

    def call(self, inputs, training=None, mask=None):
        x = self.basenet(inputs)
        x = self.conv_out(x)
        return self.spp_pool(x)


class DiagnosisModel(Model):
    def __init__(self, num_classes, basenet_weights=None, spp_net_weight=None):
        super(DiagnosisModel, self).__init__()
        self.spp_net = SPPNetModel(basenet_weights)
        if spp_net_weight is not None:
            self.spp_net.load_weights(spp_net_weight)
        self.dropout = layers.Dropout(0.3)
        self.dense_out = layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        x = self.spp_net(inputs)
        x = self.dropout(x)
        return self.dense_out(x)


class FocalLoss(losses.Loss):
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
