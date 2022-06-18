import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Dense, Activation, InputLayer
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import LeakyReLU, Multiply, Dropout


class SELayer(Model):
    def __init__(self, filters, reduction=16):
        super(SELayer, self).__init__()
        self.gap = GlobalAveragePooling2D()
        self.fc = Sequential([
            # use_bias???
            Dense(filters // reduction,
                  input_shape=(filters, ),
                  use_bias=False),
            Dropout(0.5),
            BatchNormalization(),
            Activation('relu'),
            Dense(filters, use_bias=False),
            Dropout(0.5),
            BatchNormalization(),
            Activation('sigmoid')
        ])
        self.mul = Multiply()

    def call(self, input_tensor):
        weights = self.gap(input_tensor)
        weights = self.fc(weights)
        return self.mul([input_tensor, weights])
"""
def se(inputs,filters, reduction=16):
    x = GlobalAveragePooling2D()(inputs)
    x = Dense(filters // reduction,input_shape=(filters, ),use_bias=False)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(filters, use_bias=False)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    out = Multiply()([inputs, x])
    return out
"""