import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model, regularizers
#from attention import SE
#from attention import AA
#from attention import CBAM
#from attention import ECA
#from attention import MY2
from attention import MY
#from attention import AA
def regularized_padded_conv(*args, **kwargs):
    """
    定义一个3x3卷积！kernel_initializer='he_normal','glorot_normal'
    :param args:
    :param kwargs:
    :return:
    """
    return layers.Conv2D(*args, **kwargs, padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.orthogonal(),
                         kernel_regularizer=regularizers.l2(5e-4))

class BottleNeck(layers.Layer):
    def __init__(self, growth_rate):
        super(BottleNeck, self).__init__()
        inner_channel = 4 * growth_rate

        self.bottle_neck = Sequential([
            layers.BatchNormalization(),
            layers.ReLU(),
            regularized_padded_conv(inner_channel, (1, 1)),
            layers.BatchNormalization(),
            layers.ReLU(),
            regularized_padded_conv(growth_rate, (3, 3))
        ])

    def call(self, x, training=False):
        x1 = self.bottle_neck(x, training=training)

        return tf.concat([x, x1], axis=-1)


class Transition(layers.Layer):
    def __init__(self, out_channels):
        super(Transition, self).__init__()

        self.down_sample = Sequential([
            layers.BatchNormalization(),
            regularized_padded_conv(out_channels, (1, 1)),
            layers.AveragePooling2D((2, 2), strides=2)
        ])

    def call(self, x, training=False):
        return self.down_sample(x, training=training)


class DenseNet(Model):
    def __init__(self,
                 num_classes,
                 block,
                 nblocks,
                 growth_rate=12,
                 reduction=0.5,
                 input_shape=(32, 32, 3)):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        inner_channels = 2 * growth_rate

        self.conv1 = Sequential([
            layers.Input(input_shape),
            regularized_padded_conv(inner_channels, (3, 3))
        ])

        self.features = Sequential()

        for idx in range(len(nblocks) - 1):
            self.features.add(block(nblocks[idx]))
            inner_channels += growth_rate * nblocks[idx]

            out_channels = int(reduction * inner_channels)
            self.features.add(Transition(out_channels))
            inner_channels = out_channels

        self.features.add(self._make_dense_layers(
            block, nblocks[len(nblocks)-1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add(layers.BatchNormalization())
        self.features.add(layers.ReLU())

        self.gap = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes, activation='softmax')

    def _make_dense_layers(self, block, nblocks):
        dense_block = Sequential()
        for idx in range(nblocks):
            dense_block.add(block(self.growth_rate))
        return dense_block

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.features(x, training=training)
        x = self.gap(x)
        x = self.fc(x)
        return x


def densenet121(num_classes):
    return DenseNet(num_classes, BottleNeck, [6, 12, 24, 16], growth_rate=32)


def densenet169(num_classes):
    return DenseNet(num_classes, BottleNeck, [6, 12, 32, 32], growth_rate=32)


def densenet201(num_classes):
    return DenseNet(num_classes, BottleNeck, [6, 12, 48, 32], growth_rate=32)


def densenet161(num_classes):
    return DenseNet(num_classes, BottleNeck, [6, 12, 36, 24], growth_rate=12)
    #return DenseNet(num_classes, BottleNeck, [6, 12, 36, 24], growth_rate=48)
