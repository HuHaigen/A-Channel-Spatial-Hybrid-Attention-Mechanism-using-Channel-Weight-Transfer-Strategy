import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model
from attention import SE
#from attention import AA
#from attention import CBAM
#from attention import ECA
#from attention import MY2
from attention import MY
#from attention import AA

class Inception(layers.Layer):
    def __init__(self, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super(Inception, self).__init__()

        self.b1 = Sequential([
            layers.Conv2D(n1x1, (1, 1)),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.b2 = Sequential([
            layers.Conv2D(n3x3_reduce, (1, 1)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(n3x3, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.b3 = Sequential([
            layers.Conv2D(n5x5_reduce, (1, 1)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(n5x5, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(n5x5, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])
        self.b4 = Sequential([
            layers.MaxPool2D((3, 3), 1, padding='same'),
            layers.Conv2D(pool_proj, (1, 1)),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])

    def call(self, x):
        x = tf.concat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], axis=3)

        return x


class GoogleNet(Model):
    def __init__(self, num_classes, input_shape=(32, 32, 3)):
        super(GoogleNet, self).__init__()
        self.layer1 = Sequential([
            layers.Input(input_shape),
            layers.Conv2D(192, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        #self.layer1_ac = CBAM.ChannelAttention(192)
        #self.layer1_as = CBAM.SpatialAttention()
        self.layer1_ac = MY.ChannelAttention()
        #self.layer1_as = MY.SpatialAttention(192)
        #self.layer1_ac = SE.SELayer(192)
        #self.layer1_ac = ECA.ECA()
        #self.layer1_ac = AA.DoubleAttentionLayer(192)
        self.layer2 = Sequential([
            Inception(64, 96, 128, 16, 32, 32),
            #MY.ChannelAttention(),
            #MY.SpatialAttention(256),
            Inception(128, 128, 192, 32, 96, 64),
            #MY.ChannelAttention(),
            #MY.SpatialAttention(480),
            layers.MaxPool2D((3, 3), 2, padding='same'),
        ])
        #self.layer2_ac = CBAM.ChannelAttention(480)
        #self.layer2_as = CBAM.SpatialAttention()
        self.layer2_ac = MY.ChannelAttention()
       # self.layer2_as = MY.SpatialAttention(480)
        #self.layer2_ac = SE.SELayer(480)
        #self.layer2_ac = ECA.ECA()
        #self.layer2_ac = AA.DoubleAttentionLayer(480)
        self.layer3 = Sequential([
            Inception(192, 96, 208, 16, 48, 64),
            #MY.ChannelAttention(),
            #MY.SpatialAttention(512),
            Inception(160, 112, 224, 24, 64, 64),
            #MY.ChannelAttention(),
            #MY.SpatialAttention(512),
            Inception(128, 128, 256, 24, 64, 64),
            #MY.ChannelAttention(),
            #MY.SpatialAttention(128*4),
            Inception(112, 144, 288, 32, 64, 64),
            #MY.ChannelAttention(),
            #MY.SpatialAttention(528),
            Inception(256, 160, 320, 32, 128, 128),
            #MY.ChannelAttention(),
            #MY.SpatialAttention(832),
            layers.MaxPool2D((3, 3), 2, padding='same'),
        ])
        #self.layer3_ac = CBAM.ChannelAttention(832)
        #self.layer3_as = CBAM.SpatialAttention()
        self.layer3_ac = MY.ChannelAttention()
       # self.layer3_as = MY.SpatialAttention(832)
        #self.layer3_ac = SE.SELayer(832)
        #self.layer3_ac = ECA.ECA()
        #self.layer3_ac = AA.DoubleAttentionLayer(832)
        self.layer4 = Sequential([
            Inception(256, 160, 320, 32, 128, 128),
            #MY.ChannelAttention(),
            #MY.SpatialAttention(832),
            Inception(384, 192, 384, 48, 128, 128),
            #MY.ChannelAttention(),
            #MY.SpatialAttention(1024)
        ])
        #self.layer4_ac = CBAM.ChannelAttention(1024)
        #self.layer4_as = CBAM.SpatialAttention()
        self.layer4_ac = MY.ChannelAttention()
        #self.layer4_as = MY.SpatialAttention(1024)
        #self.layer4_ac = SE.SELayer(1024)
        #self.layer4_ac = ECA.ECA()
        #self.layer4_ac = AA.DoubleAttentionLayer(1024)
        self.layer5 = Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.4),
        ])
        self.fc = layers.Dense(num_classes, activation='softmax')
        self.flat = layers.Flatten()
    def call(self, inputs, training=False):
        x = self.layer1(inputs, training=training)
        x = self.layer1_ac(x)
        #x = self.layer1_as(x)
        x = self.layer2(x, training=training)
        x = self.layer2_ac(x)
        #x = self.layer2_as(x)
        x = self.layer3(x, training=training)
        x = self.layer3_ac(x)
        #x = self.layer3_as(x)
        x = self.layer4(x, training=training)
        x = self.layer4_ac(x)
        #x = self.layer4_as(x)
        x = self.layer5(x, training=training)
        #x = tf.reshape(x, (x.shape[0], -1))
        x = self.flat(x)
        x = self.fc(x)
        return x


def GoogLeNet(num_classes):
    return GoogleNet(num_classes)
