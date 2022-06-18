import tensorflow as tf
import math
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects
class Gelu(Activation):
    def __init__(self, activation, **kwargs):
        super(Gelu, self).__init__(activation, **kwargs)
        self.__name__ = 'Gelu'

def gelu(inputs):
    cdf = 0.5 * (1.0 + tf.tanh(
        (math.sqrt(2 / math.pi) * (inputs + 0.044715 * tf.pow(inputs, 3)))))
    return inputs * cdf

get_custom_objects().update({'Gelu': Gelu(gelu)})





"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

#@tf.keras.utils.register_keras_serializable(package='Text')

#Gaussian error linear unit.
def gelu(x):
  #Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.
  Returns:
    `x` with the GELU activation applied.

  cdf = 0.5 * (1.0 + tf.tanh(
      (math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf



"""