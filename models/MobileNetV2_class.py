import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model, regularizers
#from attention import SE
#from attention import AA
#from attention import CBAM
#from attention import ECA
#from attention import MY2
from attention import MY_class as MY
#from attention import AA
def regularized_padded_conv(*args, **kwargs):
    """
    定义一个3x3卷积！kernel_initializer='he_normal','glorot_normal'
    :param args:
    :param kwargs:
    :return:
    """
    return layers.Conv2D(*args, **kwargs, padding='same', use_bias=False, kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.l2(5e-4))
def regularized_padded_depthwiseconv(*args, **kwargs):
    return layers.DepthwiseConv2D(*args, **kwargs, padding='same', use_bias=False, depthwise_initializer='he_normal',
                         depthwise_regularizer=regularizers.l2(5e-4))


def ReLU6():
    return layers.Lambda(lambda x: tf.nn.relu6(x))


class LinearBottleNeck(layers.Layer):
    def __init__(self, in_channels, out_channels, strides=1, t=6):
        super(LinearBottleNeck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides

        self.residual = Sequential([
            regularized_padded_conv(in_channels * t,
                          (1, 1),
                          strides=1),
            layers.BatchNormalization(),
            ReLU6(),
            regularized_padded_depthwiseconv((3, 3),
                                   strides=strides),
            layers.BatchNormalization(),
            ReLU6(),
            regularized_padded_conv(out_channels,
                          (1, 1),
                          strides=1),
            layers.BatchNormalization(),
        ])
        self.attention1 = MY.ChannelAttention()
        self.attention2 = MY.SpatialAttention()
        #self.attention1 = SE.SELayer(out_channels)
        #self.attention1 = ECA.ECA()
        #self.attention1 = AA.DoubleAttentionLayer(out_channels)
        #self.attention1 = CBAM.ChannelAttention(out_channels)
        #self.attention2 = CBAM.SpatialAttention()

    def call(self, x, training=False):
        residual = self.residual(x, training=training)
        residual1 = self.attention1(residual)
        residual2 = self.attention2(residual1)

        if self.strides == 1 and self.in_channels == self.out_channels:

            residual2 += x
            residual = residual2

        return residual


class MobileNetV2(Model):
    def __init__(self, num_classes, input_shape=(32, 32, 3)):
        super(MobileNetV2, self).__init__()

        self.front = Sequential([
            layers.Input(input_shape),
            layers.BatchNormalization(),
            ReLU6()
        ])
        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = regularized_padded_conv(filters=1280,
                                   kernel_size=(1, 1),
                                   strides=1)
        self.ap = layers.AveragePooling2D((7, 7),padding='same')
        self.flat = layers.Flatten()
        self.fc = layers.Dense(num_classes, activation='softmax')
    def _make_stage(self, repeat, in_channels, out_channels, strides, t):
        nets = Sequential()
        nets.add(LinearBottleNeck(in_channels, out_channels, strides, t))

        while repeat - 1:
            nets.add(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1
        return nets

    def call(self, inputs, training=False):
        x = self.front(inputs)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        print(x.shape)
        x = self.ap(x)
        x = self.flat(x)
       # x = tf.reshape(x, (x.shape[0], -1))
        x = self.fc(x)
        return x


def mobilenetv2(num_classes):
    return MobileNetV2(num_classes)
