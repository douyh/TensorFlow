# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 15:43:14 2018

@author: douyh
"""

from datetime import datetime
import math
import time
import tensorflow as tf

#设置batch
batch_size = 32
num_batches = 100

################################################################################
#conv_op
#创建卷积层并把本层的参数存入参数列表
################################################################################
def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    '''
    :param input_op:输入tensor
    :param name:本层名称
    :param kh:卷积核高
    :param kw:卷积核宽
    :param n_out:卷积核数量即输出通道数
    :param dh:步长高
    :param dw:步长宽
    :param p:参数列表
    :return:返回卷积结果
    '''
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        #初始化卷积核
        kernel = tf.get_variable(scope + 'w', shape = [kh, kw, n_in, n_out], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer_conv2d())
        #对输入进行卷积
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding = 'SAME')
        bias = tf.Variable(tf.constant(0.0, shape = [n_out], dtype = tf.float32), trainable = True, name = 'bias')
        z = tf.nn.bias_add(conv, bias)
        activation = tf.nn.relu(z, name = scope)
        p += [kernel, bias]
        return activation

################################################################################
#fc_op
#创建全连接层并把本层的参数存入参数列表
################################################################################
def fc_op(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        weights = tf.get_variable(scope + 'w', shape = [n_in, n_out], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
        bias = tf.Variable(tf.constant(0.1, shape = [n_out], dtype = tf.float32), name = 'bias')#这个函数把矩阵，加法,relu都放在了一起
        activation = tf.nn.relu_layer(input_op, weights, bias, name = scope)
        p += [weights, bias]
        return activation

################################################################################
#mpool_op
#创建最大池化层并把本层的参数存入参数列表
################################################################################
def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op, ksize = [1, kh, kw, 1], strides = [1, dh, dw, 1], padding = 'SAME', name = name)

################################################################################
#inference_op
#创建VGG16的网络结构
################################################################################
def inference_op(input_op, keep_prob):
    '''
    :param input_op: 输入tensor (224,224,3)
    :param keep_prob: 控制dropout比率的一个placeholder
    :return:
    '''
    p = []#初始化参数列表

    #第一段卷积层，通道数为64，卷积核大小为3*3，步长为1，输出为(112,112,64)
    conv1_1 = conv_op(input_op, name = 'conv1_1', kh = 3, kw = 3, n_out = 64, dh = 1, dw = 1, p = p)
    conv1_2 = conv_op(conv1_1, name ='conv1_2', kh = 3, kw = 3, n_out = 64, dh = 1, dw = 1, p = p)
    pool1 = mpool_op(conv1_2, name = 'pool1', kh = 2, kw = 2, dw = 2, dh = 2)

    #第二段卷积层，通道数为128,卷积核大小为3*3,步长为1，输出为(56,56,128)
    conv2_1 = conv_op(pool1, name = 'conv2_1', kh = 3, kw = 3, n_out = 128, dh = 1, dw = 1, p = p)
    conv2_2 = conv_op(conv2_1, name = 'conv2_2', kh = 3, kw = 3, n_out = 128, dh = 1, dw = 1, p = p)
    pool2 = mpool_op(conv2_2, name = 'pool2', kh = 2, kw = 2, dw =2, dh = 2)

    #第三段卷积层，通道数为256,卷积核大小为3*3,步长为1，输出为(28,28,256)
    conv3_1 = conv_op(pool2, name = 'conv3_1', kh = 3, kw = 3, n_out = 256, dh = 1, dw = 1, p = p)
    conv3_2 = conv_op(conv3_1, name = 'conv3_2', kh = 3, kw = 3, n_out = 256, dh = 1, dw = 1, p = p)
    pool3 = mpool_op(conv3_2, name = 'pool3', kh = 2, kw = 2, dw =2, dh = 2)

    #第四段卷积层，通道数为512,卷积核大小为3*3,步长为1，输出为(14,14,512)
    conv4_1 = conv_op(pool3, name = 'conv4_1', kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = p)
    conv4_2 = conv_op(conv4_1, name = 'conv4_2', kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = p)
    pool4 = mpool_op(conv4_2, name = 'pool4', kh = 2, kw = 2, dw =2, dh = 2)

    #第五段卷积层，通道数为512,卷积核大小为3*3,步长为1，输出为(7,7,512)
    conv5_1 = conv_op(pool4, name = 'conv5_1', kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = p)
    conv5_2 = conv_op(conv5_1, name = 'conv5_2', kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = p)
    pool5 = mpool_op(conv5_2, name = 'pool5', kh = 2, kw = 2, dw =2, dh = 2)

    #将输出结果扁平化，转化为一维向量
    shp = pool5.get_shape()#.as_list() 使用了这个就不需要.value
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name = 'resh1')
    #上述可以替换为
    #resh1 = tf.reshape(pool5, [pool5.shape[0].value, -1], name = 'resh1')

    #全连接层
    fc6 = fc_op(resh1, name = 'fc6', n_out = 4096, p = p)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name = 'fc6_drop')

    fc7 = fc_op(fc6_drop, name = 'fc7', n_out = 4096, p = p)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name = 'fc7_drop')

    fc8 = fc_op(fc7_drop, name = 'fc8', n_out = 1000, p = p)
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax, 1)

    return predictions, softmax, fc8, p

################################################################################
#评估AlexNet每轮计算时间的函数time_tensorflow_run
################################################################################
def time_tensorflow_run(session, target, feed, info_string):
    num_steps_burn_in = 10 #程序预热轮数
    total_duration = 0.0
    total_duration_squard = 0.0

    #在初始热身的10次迭代之后，每10轮迭代显示当前迭代所需要的时间，同时累积计算，以便后面计算均值和标准差
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target, feed_dict = feed)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f'%(datetime.now(), i - num_steps_burn_in, duration))#datetime.now()
            total_duration += duration
            total_duration_squard += duration * duration

    #循环结束后，计算每轮迭代的平均耗时mn和标准差sd可最后将结果显示出来，这样就完成了计算每轮迭代耗时的评测函数
    mn = total_duration / num_batches
    vr = total_duration_squard / num_batches - mn * mn#方差等于平方的均值减去均值的平方
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch'%(datetime.now(), info_string, num_batches, mn, sd))

################################################################################
#评测函数
################################################################################
def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3], dtype  = tf.float32, stddev = 1e-1))

        keep_prob = tf.placeholder(tf.float32)
        predictions, softmax, fc8, p = inference_op(images, keep_prob)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        time_tensorflow_run(sess, predictions, {keep_prob: 1.0}, 'Forward')
        objective = tf.nn.l2_loss(fc8)
        grad = tf.gradients(objective, p)
        time_tensorflow_run(sess, grad, {keep_prob: 0.5}, 'Forward-backward')

run_benchmark()
