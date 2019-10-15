import tensorflow as tf
import numpy as np
import os
from functools import reduce


def get_shape(tensor):
    return tensor.get_shape().as_list()


def flatten(tensor):
    shape = get_shape(tensor)
    return tf.reshape(tensor, [shape[0], reduce(lambda x, y: x * y, shape[1:])])

def conv2d(input, output_channels, kernal_size=3, strides=1, name=None, use_bias=False, padding='SAME',
           initializer = tf.random_normal_initializer(mean=0, stddev=0.02)):
    input_shape = get_shape(input)
    input_channels = input_shape[-1]
    filter_shape = [kernal_size, kernal_size, input_channels, output_channels]
    strides_shape = [1, strides, strides, 1]

    with tf.variable_scope(name):
        weight = tf.get_variable('weight', shape=filter_shape, initializer=initializer)
        _conv2d = tf.nn.conv2d(input, filter=weight, strides=strides_shape, padding=padding)
        if (use_bias):
            bias = tf.get_variable('bias', shape=filter_shape[-1], initializer=tf.constant_initializer(0.))
            _conv2d = tf.nn.bias_add(_conv2d, bias)

        return _conv2d


def fc(input, output_channels, name=None, use_bias=False,
       initializer=tf.random_normal_initializer(mean=0., stddev=0.02)):
    input_shape = get_shape(input)
    if (len(input_shape) == 4):
        input = flatten(input)
        input_shape = get_shape(input)

    weight_shape = [input_shape[-1], output_channels]
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', shape=weight_shape, initializer=initializer)
        _fc = tf.matmul(input)
        if (use_bias):
            bias = tf.get_variable('bias', shape=weight_shape[-1], initializer=tf.constant_initializer(0.))
            _fc = tf.nn.bias_add(_fc, bias)

        return _fc


def deconv2d(input, output_channels=None, output_size=None, name=None, kernel_size=3, strides=2, padding='SAME',use_bias = False,
             initializer=tf.random_normal_initializer(mean=0., stddev=0.02)):
    input_shape = get_shape(input)
    input_size = input_shape[1]
    input_channels = input_shape[-1]
    filter_shape = [kernel_size, kernel_size, output_channels, input_channels]
    strides_shape = [1, strides, strides, 1]
    if (not output_size):
        output_size = input_size * strides
    if (not output_channels):
        output_channels = input_channels // strides

    output_shape = [input_shape[0], output_size, output_size, output_channels]

    with tf.variable_scope(name):
        weight = tf.get_variable('weight', shape=filter_shape, initializer=initializer)
        _deconv2d = tf.nn.conv2d_transpose(input, filter=weight, strides=strides_shape, output_shape=output_shape,
                                           padding=padding)
        if(use_bias):
            bias = tf.get_variable('bias',shape = filter_shape[-1],initializer = tf.constant_initializer(0.))
            _deconv2d = tf.nn.bias_add(_deconv2d,bias)

        return _deconv2d

def maxpooling2d(input,kernel_size = 2,strides = 1,padding = 'SAME'):
    _filter = [1,kernel_size,kernel_size,1]
    _strides = [1,strides,strides,1]
    return tf.nn.max_pool(input,_filter,strides =_strides,padding = padding)

def relu(input,alpha = 0.):
    return tf.maximum(input,alpha * input)


def softmax_cross_entropy(input,label,safe_log = 1e-12,need_softmax = False):
    if(need_softmax):
        input = tf.nn.softmax(input)
    return -tf.reduce_mean( tf.reduce_sum( tf.log(input - label + safe_log), axis = 1 ), axis = 0 )

def sigmoid_cross_entropy(input,label,safe_log = 1e-12,need_sigmoid = False):
    if(need_sigmoid):
        input = tf.nn.sigmoid(input)
    return -tf.reduce_mean( tf.log( input - label + safe_log ))

def instance_normal(input,name):
    input_shape = get_shape(input)
    with tf.variable_scope(name):
        _scale = tf.get_variable('scale',input_shape[-1],initializer = tf.random_normal_initializer(mean = 0.,stddev = 0.02, dtype = tf.float32))
        _offset = tf.get_variable('offset',input_shape[-1],initializer = tf.constant_initializer(0.,dtype = tf.float32))

        mean,variance = tf.nn.moments(input,axes = [0,1,2], keep_dims = True)
        epsilon = 1e-5
        inv = ( variance + epsilon ) ** - 0.5
        normalized = (input - mean ) * inv

        return _scale * normalized + _offset

# def batch_normal(input,name = None,is_training=True, moving_decay=0.99):
#     input_shape = get_shape(input)
#
#     gamma_initializer = tf.random_normal_initializer(mean = 0, stddev = 0.02)
#     beta_initializer = tf.constant_initializer(0.)
#     with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
#         gamma = tf.get_variable('gamma',input_shape[-1],initializer = gamma_initializer)
#         beta = tf.get_variable('beta',input_shape[-1],initializer = beta_initializer)
#
#         mean,variance = tf.nn.moments(input, axes = [0,1,2], keep_dims = True)
#
#         ema = tf.train.ExponentialMovingAverage(moving_decay)
#         ema_apply_op = ema.apply([mean, variance])
#
#         def mean_vars_update():
#
#             with tf.control_dependencies([ema_apply_op]):
#                 return tf.identity(mean),tf.identity(variance)
#         if(is_training):
#             mean, variance = mean_vars_update()
#         else:
#             mean, variance = ema.average(mean), ema.average(variance)
#
#
#         epsilon = 1e-5
#         # inv = (variance + epsilon) ** -0.5
#         # normalized = (input - mean) * inv
#     # return gamma * normalized + beta
#     return tf.nn.batch_normalization(input, mean, variance, beta, gamma, epsilon)


def batch_normal(x, is_training=True, name=None, moving_decay=0.9, eps=1e-5,gamma_initializer = tf.constant_initializer(1)):

    with tf.variable_scope(name,reuse = tf.AUTO_REUSE):
        return tf.contrib.layers.batch_norm(x,
                                            decay=moving_decay,
                                            scale=True,
                                            updates_collections=None,
                                            is_training=is_training)
    # # 获取输入维度并判断是否匹配卷积层(4)或者全连接层(2)
    # shape = x.shape
    # assert len(shape) in [2, 4]
    # param_shape = shape[-1]
    # with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
    #     # 声明BN中唯一需要学习的两个参数，y=gamma*x+beta
    #
    #     # 计算当前整个batch的均值与方差
    #     axes = list(range(len(shape) - 1))
    #     batch_mean, batch_var = tf.nn.moments(x, axes, name='moments')
    #
    #     # 采用滑动平均更新均值与方差
    #     ema = tf.train.ExponentialMovingAverage(moving_decay)
    #     def mean_var_with_update():
    #         ema_apply_op = ema.apply([batch_mean, batch_var])
    #         with tf.control_dependencies([ema_apply_op]):
    #             return tf.identity(batch_mean), tf.identity(batch_var)
    #
    #     # 训练时，更新均值与方差，测试时使用之前最后一次保存的均值与方差
    #     mean, var = tf.cond(tf.equal(is_training, True) , mean_var_with_update,
    #                         lambda: (ema.average(batch_mean), ema.average(batch_var)))
    #
    #     gamma = tf.get_variable('gamma', shape = param_shape, initializer = gamma_initializer)
    #     beta = tf.get_variable('beta', shape = param_shape, initializer = tf.constant_initializer(0))
    #
    #     # 最后执行batch normalization
    #     return tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
    #
















