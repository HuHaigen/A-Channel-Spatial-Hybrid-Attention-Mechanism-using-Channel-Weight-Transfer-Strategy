import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model, regularizers
from tensorflow.keras.layers import GlobalAveragePooling2D


from attention import SE
#from attention import AA
#from attention import CBAM
#from attention import ECA
#from attention import MY_class as MY

from keras_flops import get_flops
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

def BasicBlock_Atten(inputs,kernels,stride=2):

    shortcut = regularized_padded_conv(kernels, (1, 1), strides=stride)(inputs)
    shortcut = layers.BatchNormalization()(shortcut)
    residual = shortcut


    x = regularized_padded_conv(kernels, (3, 3), strides=stride)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = regularized_padded_conv(kernels, (3, 3), strides=1)(x)
    x = layers.BatchNormalization()(x)

    #x = CBAM.ChannelAttention(kernels)(x)
    #x = CBAM.SpatialAttention()(x)
   # x = ECA.ECA()(x)
    x = SE.SELayer(kernels)(x)
    #x = AA.DoubleAttentionLayer(kernels)(x)
    #x = MY.ChannelAttention()(x)
   # x = MY.SpatialAttention()(x)

    x = x + residual
    # x = x1 + x2 + residual
    # x = x * tf.math.tanh(tf.math.softplus(x))
    x = tf.nn.relu(x)
    return x


def BasicBlock(inputs,kernels,stride=1):
    if stride != 1:
        shortcut = regularized_padded_conv(kernels, (1, 1), strides=stride)(inputs)
        shortcut = layers.BatchNormalization()(shortcut)
        residual = shortcut
    else:
        residual = inputs

    x = regularized_padded_conv(kernels, (3, 3), strides=stride)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = regularized_padded_conv(kernels, (3, 3), strides=1)(x)
    x = layers.BatchNormalization()(x)



    x = x + residual
    # x = x1 + x2 + residual
    # x = x * tf.math.tanh(tf.math.softplus(x))
    x = tf.nn.relu(x)
    return x

def BottleNeckBlock(inputs,kernels,stride=1):
    shorcut = regularized_padded_conv(kernels * 4, (1, 1), strides=stride)(inputs)
    shorcut = layers.BatchNormalization()(shorcut)

    features = regularized_padded_conv(kernels, (1, 1), strides=1)(inputs)
    features = layers.BatchNormalization()(features)
    features = regularized_padded_conv(kernels, (3, 3), strides=stride)(features)
    features = layers.BatchNormalization()(features)
    features = regularized_padded_conv(kernels * 4, (1, 1), strides=1)(features)
    features = layers.BatchNormalization()(features)

    x = features + shorcut
    # x = x1 + x2 + residual
    # x = x * tf.math.tanh(tf.math.softplus(x))
    x = tf.nn.relu(x)
    return x
def BottleNeckBlock_Atten(inputs,kernels,stride=1):
    shorcut = regularized_padded_conv(kernels * 4, (1, 1), strides=stride)(inputs)
    shorcut = layers.BatchNormalization()(shorcut)

    features = regularized_padded_conv(kernels, (1, 1), strides=1)(inputs)
    features = layers.BatchNormalization()(features)
    features = regularized_padded_conv(kernels, (3, 3), strides=stride)(features)
    features = layers.BatchNormalization()(features)
    features = regularized_padded_conv(kernels * 4, (1, 1), strides=1)(features)
    features = layers.BatchNormalization()(features)

    # x = CBAM.ChannelAttention(kernels)(x)
    # x = CBAM.SpatialAttention()(x)
    # x = ECA.ECA()(x)
    # x = SE.SELayer(kernels)(x)
    # x = AA.DoubleAttentionLayer(kernels)(x)
    #features = MY.ChannelAttention()(features)
    #features = MY.SpatialAttention()(features)

    x = features + shorcut
    # x = x1 + x2 + residual
    # x = x * tf.math.tanh(tf.math.softplus(x))
    x = tf.nn.relu(x)
    return x

def make_layerBottle(block, kernels, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    x = block
    for stride in strides:
        if stride == 1:
            x = BottleNeckBlock(x,kernels, stride)
        else:
            x = BottleNeckBlock_Atten(x, kernels, stride)
    return x

def make_layerBasic(block, kernels, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    x = block
    for stride in strides:
        if stride == 1:
            x = BasicBlock(x,kernels, stride)
        else:
            x = BasicBlock_Atten(x,kernels, stride)

    return x

def ResNetBasic(input, num_blocks):
    x = input

    x = regularized_padded_conv(64, (3, 3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = make_layerBasic(x, 64, num_blocks[0], 1)
    x = make_layerBasic(x, 128, num_blocks[1], 2)
    x = make_layerBasic(x, 256, num_blocks[2], 2)
    x = make_layerBasic(x, 512, num_blocks[3], 2)

    x = layers.GlobalAveragePooling2D()(x)
   # x = layers.Dense(num_classes, activation='softmax')(x)
    return x

def ResNetBottle(input, num_blocks):
    x = input

    x = regularized_padded_conv(64, (3, 3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = make_layerBottle(x, 64, num_blocks[0], 1)
    x = make_layerBottle(x, 128, num_blocks[1], 2)
    x = make_layerBottle(x, 256, num_blocks[2], 2)
    x = make_layerBottle(x, 512, num_blocks[3], 2)

    x = layers.GlobalAveragePooling2D()(x)
   # x = layers.Dense(num_classes, activation='softmax')(x)
    return x

def ResNet18(num_classes):
    inputs = layers.Input(shape=(224, 224, 3))

    x = ResNetBasic(inputs, [2, 2, 2, 2])
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    flops = get_flops(model, batch_size=1)
    print(f"FLOPS: {flops / 10 ** 9:.03} G")

    return model

def ResNet34(num_classes):
    inputs = layers.Input(shape=(224, 224, 3))

    x = ResNetBasic(inputs, [3, 4, 6, 3])
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    flops = get_flops(model, batch_size=1)
    print(f"FLOPS: {flops / 10 ** 9:.03} G")

    return model


def ResNet50(num_classes):
    inputs = layers.Input(shape=(224, 224, 3))

    x = ResNetBottle(inputs, [3, 4, 6, 3])
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    flops = get_flops(model, batch_size=1)
    print(f"FLOPS: {flops / 10 ** 9:.03} G")

    return model