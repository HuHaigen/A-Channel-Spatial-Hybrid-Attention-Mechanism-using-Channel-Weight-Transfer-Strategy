import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, regularizers, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Activation, Conv1D
from tensorflow.keras.layers import GlobalAveragePooling2D
import tensorflow.python.keras.backend as K


def regularized_padded_conv(*args, **kwargs):
    """
    定义一个3x3卷积！kernel_initializer='he_normal','glorot_normal'
    :param args:
    :param kwargs:
    :return:
    """
    return layers.Conv2D(*args, **kwargs, padding='same', use_bias=False, kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.l2(5e-4))


"""通道注意力机制"""


class ChannelAttention(layers.Layer):
    def __init__(self, k_size=3, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)

        self.conv1 = Conv1D(1, kernel_size=k_size, padding='same', kernel_regularizer=regularizers.l2(5e-4),
                            use_bias=True, activation=tf.nn.relu)
        self.conv2 = Conv1D(1, kernel_size=k_size, padding='same', kernel_regularizer=regularizers.l2(5e-4),
                            use_bias=True, activation=tf.nn.relu)
        # VGG使用kernel_regularizer
        # 100 0.6981
        self.conv3 = Conv1D(1, kernel_size=k_size, padding='same', kernel_regularizer=regularizers.l2(5e-4),
                            use_bias=True, activation=tf.nn.relu)
        # 0.6694
        # self.conv3 = Conv1D(1, kernel_size=k_size, padding='same', kernel_regularizer=regularizers.l2(5e-4))
        # self.conv3 = Conv1D(1, kernel_size=k_size, padding='same', activation=tf.nn.relu)

    def call(self, inputs, gamma=2, b=1, training=False):
        # print("inputs_tensor", inputs.shape)
        channels = K.int_shape(inputs)[-1]
        # t = int(abs((math.log(channels, 2) + b) / gamma))
        # k = t if t % 2 else t + 1
        x_global_avg_pool = GlobalAveragePooling2D()(inputs)
        x_global_max_pool = GlobalMaxPooling2D()(inputs)
       # print("x_global_max_pool", x_global_max_pool.shape)
        x_avgpool = Reshape((channels, 1))(x_global_avg_pool)
       # print("x_avgpool", x_avgpool.shape)
        x_maxpool = Reshape((channels, 1))(x_global_max_pool)
        avg_out = self.conv1(x_avgpool)
      #  print("avg_out", avg_out.shape)
        max_out = self.conv2(x_maxpool)

        # tf.stack([avg_out, max_out], axis=3)
        x = Concatenate(axis=2)([avg_out, max_out])
       # print("cat", x.shape)
        # print("max_out", x.shape)
        # print("spatial",spatial.shape)
        x = self.conv3(x)
        #print("X", x.shape)
        # x = avg_out + max_out
        x = Activation('sigmoid')(x)  # shape=[batch,chnnels,1]
       # print("x",x.shape)
        x = Reshape((1, 1, channels))(x)
       # print("reshap", x.shape)


        output = multiply([inputs, x])
        return output


"""空间注意力机制"""

class SpatialAttention(layers.Layer):
    def __init__(self, kernels, stride=2):
        super(SpatialAttention, self).__init__()

        self.down1 = Sequential([
           # regularized_padded_conv(kernels, (1, 1), strides=1),
            #layers.BatchNormalization(),
            layers.AveragePooling2D(),
            regularized_padded_conv(filters=1,
                          kernel_size=(3, 3),
                          strides=2),
            layers.BatchNormalization(),
            #regularized_padded_conv(kernels, (1, 1), strides=1),
            #layers.BatchNormalization(),
        ])
        """
        self.down2 = Sequential([
            #regularized_padded_conv(kernels, (1, 1), strides=1),
            #layers.BatchNormalization(),
            layers.Conv2D(filters=kernels,
                          kernel_size=(3, 3),
                          strides=1,
                         # padding="same",
                          groups=1,
                          use_bias=False, kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(5e-4)
                          ),
            layers.BatchNormalization(),
            #regularized_padded_conv(kernels, (1, 1), strides=1),
            #layers.BatchNormalization(),
        ])
        """
        self.spacital = Sequential([
            regularized_padded_conv(1, (7, 7), strides=1),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

    def call(self, inputs, training=False):

       # x = self.features(inputs, training=training)

        down1 = self.down1(inputs, training=training)
        #if down.shape != x1.shape:
        up = tf.image.resize(down1, size=(tf.shape(inputs)[1], tf.shape(inputs)[2]), method="nearest")
        #up = tf.sigmoid(up)
        #x = inputs*up


        avgpool1 = tf.reduce_mean(inputs, axis=3, keepdims=True)
        maxpool1 = tf.reduce_max(inputs, axis=3, keepdims=True)
        spatial1 = Concatenate(axis=3)([avgpool1, maxpool1])

        spatial_out = self.spacital(spatial1)+up

        #spatial_out = self.spacital(spatial1) + up1 + up2

        scale = tf.sigmoid(spatial_out)
        out = scale * inputs

        return out

