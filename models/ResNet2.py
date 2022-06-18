import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model, regularizers
from tensorflow.keras.layers import GlobalAveragePooling2D


#from attention import SE
#from attention import AA
#from attention import CBAM
#from attention import ECA
#from attention import MY2
#from attention import MY
from keras_flops import get_flops
#from attention import AA


def BasicBlock(inputs,kernels,stride=1):
    if stride != 1:
        shortcut = layers.Conv2D(kernels, (1, 1), strides=stride, padding='same')(inputs)
        shortcut = layers.BatchNormalization()(shortcut)
        residual = shortcut
    else:
        residual = inputs

    x = layers.Conv2D(kernels, (3, 3), strides=stride, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(kernels, (3, 3), strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    #x = CBAM.cbam_block(x)
    #x = ECA.eca(x)
    #x = SE.se(x,kernels)
    #x = AA.aa(x, kernels)
    #x = MY.myAttention(x)

    x = x + residual
    # x = x1 + x2 + residual
    # x = x * tf.math.tanh(tf.math.softplus(x))
    x = tf.nn.relu(x)
    return x

def BottleNeckBlock(inputs,kernels,stride=1):
    shorcut = layers.Conv2D(kernels * 4, (1, 1), strides=stride, padding='same')(inputs)
    shorcut = layers.BatchNormalization()(shorcut)

    features = layers.Conv2D(kernels, (1, 1), strides=1, padding='same')(inputs)
    features = layers.BatchNormalization()(features)
    features = layers.Conv2D(kernels, (3, 3), strides=stride, padding='same')(features)
    features = layers.BatchNormalization()(features)
    features = layers.Conv2D(kernels * 4, (1, 1), strides=1, padding='same')(features)
    features = layers.BatchNormalization()(features)

    x = features + shorcut
    # x = x1 + x2 + residual
    # x = x * tf.math.tanh(tf.math.softplus(x))
    x = tf.nn.relu(x)
    return x


def make_layerBottle(block, kernels, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    x = block
    for stride in strides:
        x = BottleNeckBlock(x,kernels, stride)
    return x

def make_layerBasic(block, kernels, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    x = block
    for stride in strides:
        x = BasicBlock(x,kernels, stride)
    return x

def ResNetBasic(input, num_blocks):
    x = input
    with tf.device("/gpu:2"):
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = make_layerBasic(x, 64, num_blocks[0], 1)
        x = make_layerBasic(x, 128, num_blocks[1], 2)
    with tf.device("/gpu:3"):
        x = make_layerBasic(x, 256, num_blocks[2], 2)
        x = make_layerBasic(x, 512, num_blocks[3], 2)

    x = layers.GlobalAveragePooling2D()(x)
   # x = layers.Dense(num_classes, activation='softmax')(x)
    return x

def ResNetBottle(input, num_blocks):
    x = input

    x = layers.Conv2D(64, (3, 3), padding='same')(x)
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