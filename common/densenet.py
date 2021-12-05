from tensorflow.keras import Model, layers


class PreDenseNet(Model):
    def __init__(self):
        super(PreDenseNet, self).__init__(name='PreDenseNet')
        self.zero_padding_3x3 = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))
        self.conv_7x7 = layers.Conv2D(64, 7, strides=2, use_bias=False)
        self.bn = layers.BatchNormalization(epsilon=1.001e-5)
        self.relu = layers.ReLU()
        self.zero_padding_1x1 = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))
        self.max_pool = layers.MaxPooling2D(3, strides=2)

    def call(self, inputs, training=None, mask=None):
        x = self.zero_padding_3x3(inputs)
        x = self.conv_7x7(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        x = self.zero_padding_1x1(x)
        return self.max_pool(x)


class ConvBlock(Model):
    def __init__(self, growth_rate, name=None):
        super(ConvBlock, self).__init__(name=name)
        self.bn1 = layers.BatchNormalization(epsilon=1.001e-5, axis=-1)
        self.relu = layers.ReLU()
        self.conv_expand = layers.Conv2D(4 * growth_rate, 3, padding='same', use_bias=False)
        self.conv_project = layers.Conv2D(growth_rate, 3, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization(epsilon=1.001e-5, axis=-1)
        self.concat = layers.Concatenate(axis=-1)

    def call(self, inputs, training=None, mask=None):
        x = self.bn1(inputs, training=training)
        x = self.relu(x)
        x = self.conv_expand(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)
        x = self.conv_project(x)
        return self.concat([inputs, x])


class DenseBlock(Model):
    def __init__(self, block, name=None):
        super(DenseBlock, self).__init__(name=name)
        self.block = block
        self.conv_blocks = [ConvBlock(growth_rate=32) for _ in range(block)]

    def call(self, x, training=None, mask=None):
        for i in range(self.block):
            x = self.conv_blocks[i](x)
        return x


class TransitionBlock(Model):
    def __init__(self):
        super(TransitionBlock, self).__init__()
        self.bn = layers.BatchNormalization(epsilon=1.001e-5, axis=-1)
        self.relu = layers.ReLU()
        self.conv = layers.Conv2D(92, 1, padding='same', use_bias=False)
        self.avg_pool = layers.AveragePooling2D(2, strides=2)

    def call(self, inputs, training=None, mask=None):
        x = self.bn(inputs, training=training)
        x = self.relu(x)
        x = self.conv(x)
        return self.avg_pool(x)


class DenseNet(Model):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.pre_densenet = PreDenseNet()
        self.dense_block_6 = DenseBlock(block=6)
        self.transition_6 = TransitionBlock()
        self.dense_block_12 = DenseBlock(block=12)
        self.transition_12 = TransitionBlock()
        self.dense_block_24 = DenseBlock(block=24)
        self.transition_24 = TransitionBlock()
        self.dense_block_16 = DenseBlock(block=16)
        self.bn = layers.BatchNormalization(epsilon=1.001e-5, axis=-1)
        self.relu = layers.ReLU()

    def call(self, x, training=None, mask=None):
        x = self.pre_densenet(x)
        x = self.dense_block_6(x)
        x = self.transition_6(x)
        x = self.dense_block_12(x)
        x = self.transition_12(x)
        x = self.dense_block_24(x)
        x = self.transition_24(x)
        x = self.dense_block_16(x)
        x = self.bn(x, training=training)
        return self.relu(x)
