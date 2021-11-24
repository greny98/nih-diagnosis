from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import densenet

from common.spp_net import SPPLayer
from nih.model import create_nih_model
from siim.configs import LABELS


def load_basenet(input_shape, weights=None):
    base_net = densenet.DenseNet201(input_shape=input_shape, include_top=False, weights='imagenet')
    if weights is not None:
        base_net.load_weights(weights).expect_partial()
    return base_net


def create_siim_model(ckpt=None):
    spp_layer = SPPLayer()
    nih_model = create_nih_model()
    nih_model.load_weights(ckpt).expect_partial()
    features = nih_model.get_layer('conv_out').output
    features = spp_layer(features)
    features = layers.Dropout(0.1)(features)
    features = layers.Dense(512, activation='relu')(features)
    features = layers.Dropout(0.25)(features)
    outputs = layers.Dense(len(LABELS), kernel_regularizer=regularizers.l2(2.5e-4))(features)
    return Model(inputs=[nih_model.input], outputs=outputs)
