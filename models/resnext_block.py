import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model, regularizers

def regularized_padded_conv(*args, **kwargs):
    """
    定义一个3x3卷积！kernel_initializer='he_normal','glorot_normal'
    :param args:
    :param kwargs:
    :return:
    """
    return layers.Conv2D(*args, **kwargs, padding='same', use_bias=False, kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.l2(5e-4))

class ResNeXt_BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filters, strides, groups):
        super(ResNeXt_BottleNeck, self).__init__()
        self.conv1 = regularized_padded_conv(filters=filters,
                                            kernel_size=(1, 1),
                                            strides=1)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.group_conv = regularized_padded_conv(filters=filters,
                                                 kernel_size=(3, 3),
                                                 strides=strides,
                                                 groups=groups)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = regularized_padded_conv(filters=2 * filters,
                                            kernel_size=(1, 1),
                                            strides=1)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.shortcut_conv = regularized_padded_conv(filters=2 * filters,
                                                    kernel_size=(1, 1),
                                                    strides=strides)
        self.shortcut_bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.group_conv(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)

        shortcut = self.shortcut_conv(inputs)
        shortcut = self.shortcut_bn(shortcut, training=training)

        output = tf.nn.relu(tf.keras.layers.add([x, shortcut]))
        return output


def build_ResNeXt_block(filters, strides, groups, repeat_num):
    block = tf.keras.Sequential()
    block.add(ResNeXt_BottleNeck(filters=filters,
                                 strides=strides,
                                 groups=groups))
    for _ in range(1, repeat_num):
        block.add(ResNeXt_BottleNeck(filters=filters,
                                     strides=1,
                                     groups=groups))

    return block