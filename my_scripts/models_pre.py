"""
Author: Zhou Chen
Date: 2019/11/4
Desc: 模型构建
"""
import tensorflow as tf


def vgg16(input_shape, num_classes):
    """
    使用预训练的VGG模型
    :param input_shape:
    :param num_classes:
    :return:
    """
    net = tf.keras.applications.VGG16(weights='imagenet', include_top=False, pooling='max')
    net.trainable = False  # 关闭预训练网络的参数训练
    vgg = tf.keras.Sequential([
        net,
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    vgg.build(input_shape=input_shape)
    print(vgg.summary())
    return vgg


def resnet50(input_shape, num_classes):
    """
    使用预训练的ResNet模型
    :param input_shape:
    :param num_classes:
    :return:
    """
    net = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, pooling='max')
    net.trainable = False  # 关闭预训练网络的参数训练
    resnet = tf.keras.Sequential([
        net,
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    resnet.build(input_shape=input_shape)
    print(resnet.summary())
    return resnet


def densenet121(input_shape, num_classes):
    """
    使用预训练的DenseNet模型
    :param input_shape:
    :param num_classes:
    :return:
    """
    net = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, pooling='max')
    net.trainable = False  # 关闭预训练网络的参数训练
    densenet = tf.keras.Sequential([
        net,
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    densenet.build(input_shape=input_shape)
    print(densenet.summary())
    return densenet
