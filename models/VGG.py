import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#from attention import SE
#from attention import AA
#from attention import CBAM
#from attention import ECA
from attention import MY

cfg = {


    'A': [64,          'A1', 'M', 128,          'A2',  'M', 256, 256, 'A3',    'M', 512, 512,'A4', 'M', 512, 512, 'A5', 'M'],
    'B': [64, 64,       'M', 128, 128,       'M', 256, 256,                  'M', 512, 512,                  'M', 512, 512,                 'M'],
    'D': [64, 64,       'M', 128, 128,       'M', 256, 256, 256,             'M', 512, 512, 512,             'M', 512, 512, 512,            'M'],
    'E': [64, 64,       'M', 128, 128,       'M', 256, 256, 256, 256,        'M', 512, 512, 512, 512,        'M', 512, 512, 512, 512,       'M']
}

class VGG(keras.Model):
    def __init__(self, features, num_classes, input_shape=(32, 32, 3)):
        super(VGG, self).__init__()

        self.features = keras.Sequential([
            layers.Input(input_shape),
            features
        ])

        self.classifier = keras.Sequential([
            layers.Flatten(),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax'),
        ])

    def call(self, inputs, training=False):
        x = self.features(inputs, training=training)
        #print("self.features",x.shape)
        x = self.classifier(x, training=training)

        return x


def make_layers(cfg):
    nets = []

    for l in cfg:
        if l == 'M':

            nets += [layers.MaxPool2D()]
            continue

        if l == 'A1':
            nets += [MY.ChannelAttention()]
            #nets += [SE.SELayer(64)]
            #nets += [ECA.ECA()]
            #nets += [CBAM.ChannelAttention(64)]
            #nets += [CBAM.SpatialAttention()]
            continue
        if l == 'A2':
            nets += [MY.ChannelAttention()]
            #nets += [SE.SELayer(128)]
            #nets += [ECA.ECA()]
            #nets += [CBAM.ChannelAttention(128)]
            #nets += [CBAM.SpatialAttention()]
            continue
        if l == 'A3':
            nets += [MY.ChannelAttention()]
            #nets += [SE.SELayer(256)]
            #nets += [ECA.ECA()]
            #nets += [CBAM.ChannelAttention(256)]
            #nets += [CBAM.SpatialAttention()]
            continue
        if l == 'A4':
            nets += [MY.ChannelAttention()]
            #nets += [SE.SELayer(512)]
            #nets += [ECA.ECA()]
            #nets += [CBAM.ChannelAttention(512)]
            #nets += [CBAM.SpatialAttention()]
            continue
        if l == 'A5':
            nets += [MY.ChannelAttention()]
            #nets += [SE.SELayer(512)]
            #nets += [ECA.ECA()]
            #nets += [CBAM.ChannelAttention(512)]
            #nets += [CBAM.SpatialAttention()]
            continue


        nets += [layers.Conv2D(l, (3, 3), padding='same')]
        nets += [layers.BatchNormalization()]

    return keras.Sequential(nets)


def VGG11(num_classes):
    return VGG(make_layers(cfg['A']), num_classes)


def VGG13(num_classes):
    return VGG(make_layers(cfg['B']), num_classes)


def VGG16(num_classes):
    return VGG(make_layers(cfg['D']), num_classes)


def VGG19(num_classes):
    return VGG(make_layers(cfg['E']), num_classes)