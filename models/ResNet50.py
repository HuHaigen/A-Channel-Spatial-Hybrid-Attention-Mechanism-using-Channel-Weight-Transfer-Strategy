import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model, regularizers
from tensorflow.keras.layers import GlobalAveragePooling2D


#from attention import SE
#from attention import AA
#from attention import CBAM
#from attention import ECA
#from attention import MY2
from attention import MY
from attention import AA



def regularized_padded_conv(*args, **kwargs):
    """
    定义一个3x3卷积！kernel_initializer='he_normal','glorot_normal'
    :param args:
    :param kwargs:
    :return:
    """
    return layers.Conv2D(*args, **kwargs, padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.orthogonal())
class BasicBlock(layers.Layer):
    def __init__(self, kernels, stride=1):
        super(BasicBlock, self).__init__()

        self.features = Sequential([
            regularized_padded_conv(kernels, (3, 3), strides=stride),
            layers.BatchNormalization(),
            layers.ReLU(),
            regularized_padded_conv(kernels, (3, 3), strides=1),
            layers.BatchNormalization()
        ])


        if stride != 1:
            shortcut = [
                regularized_padded_conv(kernels, (1, 1), strides=stride),
                layers.BatchNormalization()
            ]
        else:
            shortcut = []
        self.shorcut = Sequential(shortcut)
        #self.attention1 = MY.ChannelAttention()
        #self.attention2 = MY.SpatialAttention(kernels)
        #self.attention = SE.SELayer(kernels)
        #self.attention = ECA.ECA()
        #self.attention = AA.DoubleAttentionLayer(kernels)
        #self.attention1 = CBAM.ChannelAttention(kernels)
        #self.attention2 = CBAM.SpatialAttention()



    def call(self, inputs, training=False):

        residual = self.shorcut(inputs, training=training)
        x = self.features(inputs, training=training)
       # x1 = self.attention1(x)
        #x2 = self.attention2(x1)
        #x = self.attention(x)
        x = x + residual
        #x = x1 + x2 + residual
        #x = x * tf.math.tanh(tf.math.softplus(x))
        x = tf.nn.relu(x)
        return x


class BottleNeckBlock(layers.Layer):
    def __init__(self, kernels, stride=1):
        super(BottleNeckBlock, self).__init__()

        self.features = Sequential([
            regularized_padded_conv(kernels, (1, 1), strides=1),
            layers.BatchNormalization(),
            regularized_padded_conv(kernels, (3, 3), strides=stride),
            layers.BatchNormalization(),
            regularized_padded_conv(kernels * 4, (1, 1), strides=1),
            layers.BatchNormalization(),
        ])

        self.shorcut = Sequential([
            regularized_padded_conv(kernels * 4, (1, 1), strides=stride),
            layers.BatchNormalization()
        ])
       # self.attention1 = MY.ChannelAttention()
        #self.attention2 = MY.SpatialAttention(kernels*4)
        #self.attention = SE.SELayer(kernels*4)
        #self.attention = ECA.ECA()
        #self.attention = AA.DoubleAttentionLayer(kernels*4)
        #self.attention1 = CBAM.ChannelAttention(kernels)
        #self.attention2 = CBAM.SpatialAttention()

    def call(self, inputs, training=False):

        #print("inputs",inputs.shape)
        residual = self.shorcut(inputs, training=training)
        #print("residual", residual.shape)
        x = self.features(inputs, training=training)
        #x1 = self.attention1(x)
        #x2 = self.attention2(x1)
        #x = self.attention(x)
        x = x + residual
        #x = x1 + x2 + residual
        #x = x * tf.math.tanh(tf.math.softplus(x))
        x = tf.nn.relu(x)

        return x


class ResNet(Model):
    def __init__(self, block, num_blocks, num_classes, input_shape=(32, 32, 3)):
        super(ResNet, self).__init__()
        with tf.device("/gpu:1"):
            self.conv1 = Sequential([
                layers.Input(input_shape),
                regularized_padded_conv(64, (3, 3)),
                layers.BatchNormalization(),
                layers.ReLU()
            ])
           # self.attention_conv1_1 = CBAM.ChannelAttention(64)
            #self.attention_conv1_2 = CBAM.SpatialAttention()
            #self.attention_conv1_1 = MY.ChannelAttention()
            #self.attention_conv1_2 = MY.SpatialAttention(64)


            self.conv2_x = self._make_layer(block, 64, num_blocks[0], 1)
            #self.attention_conv2_1 = CBAM.ChannelAttention(64)
            #self.attention_conv2_2 = CBAM.SpatialAttention()
            #self.attention_conv2_1 = MY.ChannelAttention()
            #self.attention_conv2_2 = MY.SpatialAttention(64)

        #with tf.device("/gpu:2"):
            self.conv3_x = self._make_layer(block, 128, num_blocks[1], 2)
           # self.attention_conv3_1 = CBAM.ChannelAttention(128)
           # self.attention_conv3_2 = CBAM.SpatialAttention()
            #self.attention_conv3_1 = MY.ChannelAttention()
            #self.attention_conv3_2 = MY.SpatialAttention(128)
       # with tf.device("/gpu:3"):
            self.conv4_x = self._make_layer(block, 256, num_blocks[2], 2)
            #self.attention_conv4_1 = CBAM.ChannelAttention(256)
            #self.attention_conv4_2 = CBAM.SpatialAttention()
           # self.attention_conv4_1 = MY.ChannelAttention()
           # self.attention_conv4_2 = MY.SpatialAttention(256)
       # with tf.device("/gpu:0"):
            self.conv5_x = self._make_layer(block, 512, num_blocks[3], 2)
            #self.attention_conv5_1 = CBAM.ChannelAttention(512)
            #self.attention_conv5_2 = CBAM.SpatialAttention()
            #self.attention_conv5_1 = MY.ChannelAttention()
           # self.attention_conv5_2 = MY.SpatialAttention(512)

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
        #x = self.attention_conv1_1(x)
        #x = self.attention_conv1_2(x)
        x = self.conv2_x(x)
       # x = self.attention_conv2_1(x)
       # x = self.attention_conv2_2(x)
        x = self.conv3_x(x)
       # x = self.attention_conv3_1(x)
       # x = self.attention_conv3_2(x)
        x = self.conv4_x(x)
        #x = self.attention_conv4_1(x)
        #x = self.attention_conv4_2(x)
        x = self.conv5_x(x)
       # x = self.attention_conv5_1(x)
       # x = self.attention_conv5_2(x)
        x = self.gap(x)
        x = self.fc(x)

        #print("x.shape", x.shape)
        return x



def ResNet18(num_classes):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    return model


def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(num_classes):
    return ResNet(BottleNeckBlock, [3, 4, 6, 3], num_classes)


def ResNet101(num_classes):
    return ResNet(BottleNeckBlock, [3, 4, 23, 3], num_classes)


def ResNet152(num_classes):
    return ResNet(BottleNeckBlock, [3, 8, 36, 3], num_classes)
