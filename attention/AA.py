from tensorflow.keras.layers import *
from tensorflow.keras.layers import Activation, Conv2D, Softmax
from tensorflow.keras.layers import GlobalAveragePooling2D
import tensorflow.python.keras.backend as K
import tensorflow as tf
from tensorflow.keras import layers




class DoubleAttentionLayer(layers.Layer):
    #def __init__(self, in_channels, c_m, c_n,k =1 ):
    def __init__(self, in_channels,k=1):
        super(DoubleAttentionLayer, self).__init__()

        self.K = k
        self.c_m = in_channels
        self.c_n = in_channels
        self.softmax = Softmax()
        self.in_channels = in_channels

        self.convA = Conv2D(self.c_m, kernel_size=1)
        self.convB = Conv2D(self.c_n, kernel_size=1)
        self.convV = Conv2D(self.c_n, kernel_size=1)

    def forward(self, x):

        b, h, w, c = tf.shape(x)

        assert c == self.in_channels,'input channel not equal!'
        #assert b//self.K == self.in_channels, 'input channel not equal!'

        A = self.convA(x)
        B = self.convB(x)
        V = self.convV(x)

        batch = int(b/self.K)

        tmpA = tf.reshape(tf.transpose(tf.reshape(A, shape=(batch,self.c_m, h*w,self.K)),perm=(0,3,2,1)),shape=(batch, self.K*h*w, self.c_m))
        tmpB = tf.reshape(tf.transpose(tf.reshape(B, shape=(batch,self.c_n, h*w,self.K)),perm=(0,3,2,1)),shape=(batch*self.c_n, self.K*h*w))
        tmpV = tf.reshape(tf.transpose(tf.reshape(V, shape=(batch,self.c_n, h*w,self.K)),perm=(0,2,1,3)),shape=(int(b*h*w), self.c_n))

        softmaxB = tf.transpose(tf.reshape(self.softmax(tmpB),shape=(batch, self.c_n, self.K*h*w)),perm=( 0, 2, 1)) #batch, self.K*h*w, self.c_n
        softmaxV = tf.transpose(tf.reshape(self.softmax(tmpV), shape=(batch, self.K*h*w, self.c_n)), perm=(0, 2, 1))


        tmpG = tmpA * softmaxB    #batch, self.c_m, self.c_n
        tmpZ = tmpG * softmaxV  #batch, self.c_m, self.K*h*w
        tmpZ = tf.reshape(tf.transpose(tf.reshape(tmpZ, shape=(batch, self.K,h*w, self.c_m)),perm=( 0, 3, 2,1), shape=(int(b), h, w,  self.c_m)))

        return tmpZ
"""
def aa(inputs, kernels):
    return DoubleAttentionLayer(kernels)(inputs)


if __name__ == "__main__":


    # tmp1        = torch.ones(2,2,3)
    # tmp1[1,:,:] = tmp1[1,:,:]*2
    # tmp2 = tmp1.permute(0,2,1)
    # print(tmp1)
    # print( tmp2)
    # print( tmp1.matmul(tmp2))

    in_channels = 10
    c_m = 4
    c_n = 3

    doubleA = DoubleAttentionLayer(in_channels, c_m, c_n)

    x = tf.Variable(tf.ones(2,6,8,in_channels))
    tmp = doubleA(x)

    print("result")
"""