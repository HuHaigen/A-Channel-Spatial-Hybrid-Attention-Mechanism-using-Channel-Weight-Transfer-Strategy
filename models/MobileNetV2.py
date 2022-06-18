import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model, regularizers
from keras_flops import get_flops
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

def make_stage(inputs, repeat, in_channels, out_channels, strides, t):
    x = LinearBottleNeck(inputs, in_channels, out_channels, strides, t)


    while repeat - 1:
        x = LinearBottleNeck(x, out_channels, out_channels, 1, t)
        repeat -= 1
    return x

def LinearBottleNeck(inputs, in_channels, out_channels, strides=1, t=6):
    residual = regularized_padded_conv(in_channels * t,(1, 1),strides=1)(inputs)
    residual = layers.BatchNormalization()(residual)
    residual = tf.nn.relu6(residual)
    residual = regularized_padded_depthwiseconv((3, 3),strides=strides)(residual)
    residual = layers.BatchNormalization()(residual)
    residual = tf.nn.relu6(residual)
    residual = regularized_padded_conv(out_channels,(1, 1), strides=1)(residual)
    residual = layers.BatchNormalization()(residual)

   # attention1 = MY.ChannelAttention()(residual)
   # attention2 = MY.SpatialAttention()(attention1)

    #if strides == 1 and in_channels == out_channels:
     #   residual = attention2 + inputs

    return residual


def Mobile(inputs):
    x = layers.BatchNormalization()(inputs)
    x = tf.nn.relu6(x)

    stage1 = LinearBottleNeck(x, 32, 16, 1, 1)
    stage2 = make_stage(stage1, 2, 16, 24, 2, 6)
    stage3 = make_stage(stage2, 3, 24, 32, 2, 6)
    stage4 = make_stage(stage3, 4, 32, 64, 2, 6)
    stage5 = make_stage(stage4, 3, 64, 96, 1, 6)
    stage6 = make_stage(stage5, 3, 96, 160, 1, 6)
    stage7 = LinearBottleNeck(stage6, 160, 320, 1, 6)

    con1 = regularized_padded_conv(filters=1280, kernel_size=(1, 1), strides=1)(stage7)


    ap = layers.AveragePooling2D((7, 7), padding='same')(con1)
    flat = layers.Flatten()(ap)

    return flat

def MobileNetV2(num_classes):
    inputs = layers.Input(shape=(224, 224, 3))
    x = Mobile(inputs)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    flops = get_flops(model, batch_size=1)
    print(f"FLOPS: {flops / 10 ** 9:.03} G")

    return model
