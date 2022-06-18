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
def myChannelAttention(inputs):
    channels = K.int_shape(inputs)[-1]
    # t = int(abs((math.log(channels, 2) + b) / gamma))
    # k = t if t % 2 else t + 1
    x_global_avg_pool = GlobalAveragePooling2D()(inputs)
    x_global_max_pool = GlobalMaxPooling2D()(inputs)
    # print("x_global_max_pool", x_global_max_pool.shape)
    x_avgpool = Reshape((channels, 1))(x_global_avg_pool)
    # print("x_avgpool", x_avgpool.shape)
    x_maxpool = Reshape((channels, 1))(x_global_max_pool)
    avg_out = Conv1D(1, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(5e-4),
                            use_bias=True)(x_avgpool)#, activation=tf.nn.relu
    #  print("avg_out", avg_out.shape)
    max_out = Conv1D(1, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(5e-4),
                            use_bias=True)(x_maxpool)#, activation=tf.nn.relu

    # tf.stack([avg_out, max_out], axis=3)
    x = Concatenate(axis=2)([avg_out, max_out])
    # print("cat", x.shape)
    # print("max_out", x.shape)
    # print("spatial",spatial.shape)
   # x = Conv1D(1, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(5e-4),
    #                        use_bias=True, activation=tf.nn.relu)(x)
    x = Conv1D(1, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(5e-4),
                            use_bias=True)(x)
    # print("X", x.shape)
    # x = avg_out + max_out
    x = Activation('sigmoid')(x)  # shape=[batch,chnnels,1]
    # print("x",x.shape)
    x = Reshape((1, 1, channels))(x)
    # print("reshap", x.shape)

    output = multiply([inputs, x])
    return output

def mySpatialAttention(inputs):
  #  channels = K.int_shape(inputs)[-1]

    down1 = layers.AveragePooling2D()(inputs)
    down1 = regularized_padded_conv(filters=1,
                          kernel_size=(3, 3),
                          strides=2)(down1)
    down1 = layers.BatchNormalization()(down1)
    # if down.shape != x1.shape:
    up = tf.image.resize(down1, size=(tf.shape(inputs)[1], tf.shape(inputs)[2]), method="nearest")
    # up = tf.sigmoid(up)
    # x = inputs*up

    avgpool1 = tf.reduce_mean(inputs, axis=3, keepdims=True)
    maxpool1 = tf.reduce_max(inputs, axis=3, keepdims=True)
    spatial1 = Concatenate(axis=3)([avgpool1, maxpool1])

    spatial_out = regularized_padded_conv(1, (7, 7), strides=1)(spatial1)
    spatial_out = layers.BatchNormalization()(spatial_out)
    #spatial_out = layers.ReLU()(spatial_out)
    spatial_out = spatial_out + up
    #print("spatial_out",spatial_out.shape)


    # spatial_out = self.spacital(spatial1) + up1 + up2

    scale = tf.sigmoid(spatial_out)
    out = scale * inputs

    return out


def myAttention(input):
    ca = myChannelAttention(input)

    sa = mySpatialAttention(ca)
    out = sa
    return out