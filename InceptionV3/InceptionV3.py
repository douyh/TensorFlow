# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 19:41:40 2018

@author: douyh
"""

from datetime import datetime
import math
import time
import tensorflow as tf

slim = tf.contrib.slim
#产生截断的正态分布
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

################################################################################
#inception_v3_arg_scope
#用于生成网络中经常用到的函数的默认参数
################################################################################
def inception_v3_arg_scope(weight_decay = 0.00004, stddev = 0.1, batch_norm_var_collection = 'moving_vars'):
    '''
    :param weight_decay:
    :param stddev:
    :param batch_norm_var_collection:
    :return:
    '''
    batch_norm_params = {
                         'decay': 0.9997,
                         'epsilon':0.001,
                         'updates_collections':tf.GraphKeys.UPDATE_OPS,
                         'variables_collections':
                             {
                                'beta':None,
                                'gamma':None,
                                'moving_mean':[batch_norm_var_collection],
                                'moving_variance':[batch_norm_var_collection],
                             }
                         }

    #slim.arg_scope是一个非常有用的工具，用来给函数的参数自动赋予某些默认值。
    #使用了slim.arg_scope后就不需要每次都重复设置参数了，只需要在有修改时设置。
    with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer = slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d], weights_initializer = tf.truncated_normal_initializer(stddev = stddev),
                            activation_fn = tf.nn.relu, normalizer_fn = slim.batch_norm, normalizer_params = batch_norm_params) as sc:
            return sc

################################################################################
#inception_v3_base
#生成Inception V3网络的卷积部分
################################################################################
def inception_v3_base(inputs, scope = None):
    '''
    :param inputs: 输入的图片数据
    :param scope: 包含了函数默认参数的环境
    :return:
    '''
    #建立一个字典表，用来保存某些关键节点供之后使用
    end_points = {}

    with tf.variable_scope(scope, 'InceptionV3', [inputs]):
        # 非Inception Module的卷积层
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride = 1, padding = 'VALID'):
            net = slim.conv2d(inputs, 32, [3, 3], stride = 2, scope = 'Conv1_3x3')#第二个参数为输出通道数
            net = slim.conv2d(net, 32, [3, 3], scope = 'Conv2_3x3')
            net = slim.conv2d(net, 64, [3, 3], padding = 'SAME', scope = 'Conv3_3x3')
            net = slim.max_pool2d(net, [3, 3], stride = 2, scope = 'MaxPool1_3x3')
            net = slim.conv2d(net, 80, [3, 3], scope = 'Conv4_3x3' )
            net = slim.conv2d(net, 192, [3, 3], stride = 2, scope = 'Conv5_3x3')
            net = slim.conv2d(net, 288, [3, 3], padding = 'SAME', scope = 'Conv6_3x3')

        #连续3个Inception组，各个组内有多个Inception Module
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride = 1, padding = 'SAME'):

            with tf.variable_scope('Inception_module_1a'):
                with tf.variable_scope('Branch0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope = 'Conv2d_0b_5x5')
                with tf.variable_scope('Branch2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0c_3x3')
                with tf.variable_scope('Branch3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope = 'Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)#在第三个维度即输出通道上合并

            with tf.variable_scope('Inception_module_1b'):
                with tf.variable_scope('Branch0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope = 'Conv2d_0b_5x5')
                with tf.variable_scope('Branch2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0c_3x3')
                with tf.variable_scope('Branch3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope = 'Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)#在第三个维度即输出通道上合并

            with tf.variable_scope('Inception_module_1c'):
                with tf.variable_scope('Branch0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)  # 在第三个维度即输出通道上合并

            with tf.variable_scope('Inception_module_2a'):
                with tf.variable_scope('Branch0'):
                    branch_0 = slim.conv2d(net, 384, [3, 3], stride = 2, padding = 'VALID', scope = 'Conv2d_1a_1x1')
                with tf.variable_scope('Branch1'):
                    branch_1 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope = 'Conv2d_0b_3x3')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride = 2, padding = 'VALID', scope = 'Conv2d_1a_1x1')
                with tf.variable_scope('Branch2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride = 2, padding = 'VALID', scope = 'MaxPool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3)

            with tf.variable_scope('Inception_module_2b'):
                with tf.variable_scope('Branch0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch1'):
                    branch_1 = slim.conv2d(net, 128, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope = 'Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope = 'Conv2d_0c_7x1')
                with tf.variable_scope('Branch2'):
                    branch_2 = slim.conv2d(net, 128, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope = 'Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope = 'Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope = 'Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope = 'Conv2d_0e_1x7')
                with tf.variable_scope('Branch3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope = 'Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            with tf.variable_scope('Inception_module_2c'):
                with tf.variable_scope('Branch0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope = 'Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope = 'Conv2d_0c_7x1')
                with tf.variable_scope('Branch2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope = 'Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope = 'Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope = 'Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope = 'Conv2d_0e_1x7')
                with tf.variable_scope('Branch3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope = 'Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            with tf.variable_scope('Inception_module_2d'):
                with tf.variable_scope('Branch0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope = 'Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope = 'Conv2d_0c_7x1')
                with tf.variable_scope('Branch2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope = 'Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope = 'Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope = 'Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope = 'Conv2d_0e_1x7')
                with tf.variable_scope('Branch3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope = 'Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            with tf.variable_scope('Inception_module_2e'):
                with tf.variable_scope('Branch0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope = 'Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope = 'Conv2d_0c_7x1')
                with tf.variable_scope('Branch2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope = 'Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope = 'Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope = 'Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope = 'Conv2d_0e_1x7')
                with tf.variable_scope('Branch3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope = 'Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            end_points['Inception module 2e'] = net#作为auxiliary classifier辅助模型的分类

            with tf.variable_scope('Inception_module_3a'):
                with tf.variable_scope('Branch0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride = 2, padding = 'VALID', scope = 'Conv2d_1a_3x3')
                with tf.variable_scope('Branch1'):
                    branch_1 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope = 'Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope = 'Conv2d_0c_7x1')
                    branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride = 2, padding = 'VALID', scope = 'Conv2d_1a_3x3')
                with tf.variable_scope('Branch2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride = 2, padding = 'VALID', scope = 'MaxPool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3)

            with tf.variable_scope('Inception_module_3b'):
                with tf.variable_scope('Branch0'):
                    branch_0 = slim.conv2d(net, 320, [1, 1], scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch1'):
                    branch_1 = slim.conv2d(net, 384, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = tf.concat([slim.conv2d(branch_1, 384, [1, 3], scope = 'Conv2d_0b_1x3'),
                                          slim.conv2d(branch_1, 384, [3, 1], scope = 'Conv2d_0b_3x1')],3)
                with tf.variable_scope('Branch2'):
                    branch_2 = slim.conv2d(net, 448, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope = 'Conv2d_0b_3x3')
                    branch_2 = tf.concat([slim.conv2d(branch_2, 384, [1, 3], scope = 'Conv2d_0c_1x3'),
                                          slim.conv2d(branch_2, 384, [3, 1], scope = 'Conv2d_0c_3x1')], 3)
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope = 'Covn2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            with tf.variable_scope('Inception_module_3c'):
                with tf.variable_scope('Branch0'):
                    branch_0 = slim.conv2d(net, 320, [1, 1], scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch1'):
                    branch_1 = slim.conv2d(net, 384, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = tf.concat([slim.conv2d(branch_1, 384, [1, 3], scope = 'Conv2d_0b_1x3'),
                                          slim.conv2d(branch_1, 384, [3, 1], scope = 'Conv2d_0b_3x1')],3)
                with tf.variable_scope('Branch2'):
                    branch_2 = slim.conv2d(net, 448, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope = 'Conv2d_0b_3x3')
                    branch_2 = tf.concat([slim.conv2d(branch_2, 384, [1, 3], scope = 'Conv2d_0c_1x3'),
                                          slim.conv2d(branch_2, 384, [3, 1], scope = 'Conv2d_0c_3x1')], 3)
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope = 'Covn2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            print(net.get_shape().as_list())
            return net, end_points

################################################################################
#inception_v3
#全局平均池化，softmax和auxiliary logits
################################################################################
def inception_v3(inputs, num_classes = 1000, is_training = True, dropout_keep_prob = 0.8,
                 prediciton_fn = slim.softmax, spatial_squeeze = True, reuse = None, scope = 'InceptionV3'):
    '''
    :param inputs: 输入
    :param num_classes:最后 需要分类的数量
    :param is_training: 标志是否是训练过程，BN和dropout只有在训练的时候才启用
    :param dropout_keep_prob: dropout时保留节点的比例
    :param prediciton_fn: 用来分类的函数
    :param spatial_squeeze: 是否对输出进行squeeze操作，去除维数为1的维度
    :param reuse:
    :param scope:
    :return:
    '''
    with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes], reuse = reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training = is_training):
            net, end_points = inception_v3_base(inputs, scope = scope)

    #处理辅助分类节点
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride = 1, padding = 'SAME'):
        aux_logits = end_points['Inception module 2e']
        with tf.variable_scope('Auxlogits'):
            aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride = 3, padding = 'VALID', scope = 'AvgPool_1a_5x5')
            aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope = 'Conv2d_1b_1x1')
            aux_logits = slim.conv2d(aux_logits, 768, [5, 5], weights_initializer = trunc_normal(0.01),
                                     padding = 'VALID', scope = 'Conv2d_2a_5x5')
            aux_logits = slim.conv2d(aux_logits, num_classes, [1, 1], activation_fn = None, normalizer_fn = None,
                                     weights_initializer = trunc_normal(0.001), scope = 'Conv2d_2b_1x1')
            if spatial_squeeze:
                aux_logits = tf.squeeze(aux_logits, [1, 2], name = 'SpatialSqueeze')
            end_points['Auxlogits'] = aux_logits

    #处理正常分类预测的逻辑
        with tf.variable_scope('Logits'):
            net = slim.avg_pool2d(net, [8, 8], padding = 'VALID', scope = 'AvgPool_1a_8x8')
            net = slim.dropout(net, keep_prob = dropout_keep_prob, scope = 'Dropout_1b')
            end_points['PreLogits'] = net
            logits = slim.conv2d(net, num_classes, [1, 1], activation_fn = None, normalizer_fn = None, scope = 'Conv2d_1c_1x1')
            if spatial_squeeze:
                logits = tf.squeeze(logits, [1, 2], name = 'SpatialSqueeze')
        end_points['Logits'] = logits
        end_points['Predictions'] = prediciton_fn(logits, scope = 'Predicitoins')

    return logits, end_points

################################################################################
#评估每轮计算时间的函数time_tensorflow_run
#input
################################################################################
def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10 #程序预热轮数
    total_duration = 0.0
    total_duration_squard = 0.0

    #在初始热身的10次迭代之后，每10轮迭代显示当前迭代所需要的时间，同时累积计算，以便后面计算均值和标准差
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
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

batch_size = 32
height, width = 299, 299
inputs = tf.random_uniform((batch_size, height, width, 3))
with slim.arg_scope(inception_v3_arg_scope()):
    logits, end_points = inception_v3(inputs, is_training = False)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
num_batches = 100
time_tensorflow_run(sess, logits, 'Forward')