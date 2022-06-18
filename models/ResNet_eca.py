import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import tensorflow.python.keras.backend as K
from activation_function.AB_eca import Activation_block

class BasicBlock(layers.Layer):
    def __init__(self, kernels, stride=1):
        super(BasicBlock, self).__init__()

        self.features = Sequential([
            layers.Conv2D(kernels, (3, 3), strides=stride, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(kernels, (3, 3), strides=1, padding='same'),
            layers.BatchNormalization()
        ])

        
        if stride != 1:
            shortcut = [
                layers.Conv2D(kernels, (1, 1), strides=stride),
                layers.BatchNormalization()
            ]
        else:
            shortcut = []
        self.shorcut = Sequential(shortcut)

    def call(self, inputs, training=False):
        residual = self.shorcut(inputs, training=training)
        x = self.features(inputs, training=training)

        x = x + residual
        #x = x * tf.tanh(tf.math.softplus(x))


        x = tf.nn.relu(layers.add([residual, x]))
        return x


class BottleNeckBlock(layers.Layer):
    def __init__(self, kernels, stride=1):
        super(BottleNeckBlock, self).__init__()



        self.features = Sequential([
            layers.Conv2D(kernels, (1, 1), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(kernels, (3, 3), strides=stride, padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(kernels * 4, (1, 1), strides=1, padding='same'),
            layers.BatchNormalization(),
        ])

        self.shorcut = Sequential([
            layers.Conv2D(kernels * 4, (1, 1), strides=stride),
            layers.BatchNormalization()
        ])


    def call(self, inputs, training=False):
        residual = self.shorcut(inputs, training=training)
        x = self.features(inputs, training=training)


        x = self.activ(x)
        out = x + residual
        """ 
        #print("x", x.shape)
        channels = K.int_shape(x)[-1]



        # t = int(abs((math.log(channels, 2) + b) / gamma))
        # k = t if t % 2 else t + 1
        x_global_avg_pool = GlobalAveragePooling2D()(x)
        scale = layers.Reshape((channels, 1))(x_global_avg_pool)
        scale = self.conv(scale)
        scale = layers.Activation('sigmoid')(scale)  # shape=[batch,chnnels,1]
        scale = layers.Reshape((1, 1, channels))(scale)

        # output = multiply([inputs_tensor,x])
        # return output
        #print(" scale", scale.shape)
        out = tf.maximum(scale * x + self.d, self.a * x + self.c)

        
        #print("out", out.shape)
        #x = tf.nn.relu(x + residual)

        #out = Activation_block()(x)
        """
        return out


class ResNet(Model):
    def __init__(self, block, num_blocks, num_classes, input_shape=(32, 32, 3)):
        super(ResNet, self).__init__()
        self.conv1 = Sequential([
            layers.Input(input_shape),
            layers.Conv2D(64, (3, 3), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.conv2_x = self._make_layer(block, 64, num_blocks[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_blocks[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_blocks[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_blocks[3], 2)
        self.gap = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes, activation='softmax')

    def _make_layer(self, block, kernels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        nets = []
        for stride in strides:
            nets.append(block(kernels, stride))
        return Sequential(nets)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.gap(x)
        x = self.fc(x)
        print("x.shape",x.shape)
        return x


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(num_classes):
    return ResNet(BottleNeckBlock, [3, 4, 6, 3], num_classes)


def ResNet101(num_classes):
    return ResNet(BottleNeckBlock, [3, 4, 23, 3], num_classes)


def ResNet152(num_classes):
    return ResNet(BottleNeckBlock, [3, 8, 36, 3], num_classes)
