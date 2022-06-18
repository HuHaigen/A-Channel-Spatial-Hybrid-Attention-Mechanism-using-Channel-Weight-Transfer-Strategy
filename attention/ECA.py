from tensorflow.keras.layers import *
from tensorflow.keras.layers import Activation, Conv1D
from tensorflow.keras.layers import GlobalAveragePooling2D
import tensorflow.python.keras.backend as K
import tensorflow as tf


from tensorflow.keras import layers, Sequential

class ECA(layers.Layer):

    def __init__(self, k_size=3, **kwargs):
        super(ECA, self).__init__(**kwargs)

        self.conv = Conv1D(1,kernel_size = k_size, padding='same')

    def call(self,inputs,gamma=2,b=1):
        #print("inputs_tensor",inputs.shape)
        channels = K.int_shape(inputs)[-1]
        #t = int(abs((math.log(channels, 2) + b) / gamma))
        #k = t if t % 2 else t + 1
        x_global_avg_pool = GlobalAveragePooling2D()(inputs)
        x = Reshape((channels, 1))(x_global_avg_pool)
        x = self.conv(x)
        x = Activation('sigmoid')(x)  # shape=[batch,chnnels,1]
        x = Reshape((1, 1, channels))(x)

        output = multiply([inputs,x])
        return output


        return out
"""
def eca(inputs):
    channels = K.int_shape(inputs)[-1]
    # t = int(abs((math.log(channels, 2) + b) / gamma))
    # k = t if t % 2 else t + 1
    x_global_avg_pool = GlobalAveragePooling2D()(inputs)
    x = Reshape((channels, 1))(x_global_avg_pool)
    x = Conv1D(1, kernel_size=3, padding='same')(x)
    x = Activation('sigmoid')(x)  # shape=[batch,chnnels,1]
    x = Reshape((1, 1, channels))(x)

    output = multiply([inputs, x])
    return output
"""