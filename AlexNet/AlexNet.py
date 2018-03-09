# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 13:20:24 2018
reference:《TensorFlow实战》--黄文坚
@author: douyh
"""
#导入系统库
from datetime import datetime
import math
import time
import tensorflow as tf

#设置batch
batch_size = 32
num_batches = 100

#定义一个函数，显示每一个卷积层或者池化层输出tensor的尺寸
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())#############################

################################################################################
#inference
#input:image
#output:pool5 parameters
################################################################################
def inference(images):
    parameters = []
    with tf.name_scope('conv1') as scope:
        kernel1 = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype = tf.float32, stddev = 1e-1), name = 'weights')#7*7的滤波器
        conv = tf.nn.conv2d(images, kernel1, [1, 4, 4, 1], padding = 'SAME')#stride = 4
        bias1 = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32), trainable = True, name = 'bias')
        bias = tf.nn.bias_add(conv, bias1)#直接用加法就可以了
        conv1 = tf.nn.relu(bias, name = scope)
        print_activations(conv1)
        parameters += [kernel1, bias1]

        lrn1 = tf.nn.lrn(conv1, 4, bias = 1.0, alpha = 0.001 / 9, beta = 0.75, name = 'lrn1')#按照论文中结构来的，规范化。其实没什么效果，而且很浪费时间
        pool1 = tf.nn.max_pool(lrn1, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool1')
        print_activations(pool1)

    with tf.name_scope('conv2') as scope:
        kernel2 = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=1e-1), name='weights')  # 5*5的滤波器
        conv = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding='SAME')#stride = 1
        bias2 = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name='bias')
        bias = tf.nn.bias_add(conv, bias2)  # 直接用加法就可以了
        conv2 = tf.nn.relu(bias, name=scope)
        print_activations(conv2)
        parameters += [kernel2, bias2]

        lrn2 = tf.nn.lrn(conv2, 4, bias = 1.0, alpha = 0.001 / 9, beta = 0.75, name = 'lrn2')#按照论文中结构来的，规范化。其实没什么效果
        pool2 = tf.nn.max_pool(lrn2, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool2')
        print_activations(pool2)

    with tf.name_scope('conv3') as scope:
        kernel3 = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype = tf.float32, stddev = 1e-1), name = 'weights')#3*3的滤波器
        conv = tf.nn.conv2d(pool2, kernel3, [1, 1, 1, 1], padding = 'SAME')#stride = 1
        bias3 = tf.Variable(tf.constant(0.0, shape = [384], dtype = tf.float32), trainable = True, name = 'bias')
        bias = tf.nn.bias_add(conv, bias3)#直接用加法就可以了
        conv3 = tf.nn.relu(bias, name = scope)
        print_activations(conv3)
        parameters += [kernel3, bias3]

    with tf.name_scope('conv4') as scope:
        kernel4 = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype = tf.float32, stddev = 1e-1), name = 'weights')#3*3的滤波器
        conv = tf.nn.conv2d(conv3, kernel4, [1, 1, 1, 1], padding = 'SAME')#stride = 1
        bias4 = tf.Variable(tf.constant(0.0, shape = [256], dtype = tf.float32), trainable = True, name = 'bias')
        bias = tf.nn.bias_add(conv, bias4)#直接用加法就可以了
        conv4 = tf.nn.relu(bias, name = scope)
        print_activations(conv4)
        parameters += [kernel4, bias4]

    with tf.name_scope('conv5') as scope:
        kernel5 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype = tf.float32, stddev = 1e-1), name = 'weights')#3*3的滤波器
        conv = tf.nn.conv2d(conv4, kernel5, [1, 1, 1, 1], padding = 'SAME')#stride = 1
        bias5 = tf.Variable(tf.constant(0.0, shape = [256], dtype = tf.float32), trainable = True, name = 'bias')
        bias = tf.nn.bias_add(conv, bias5)#直接用加法就可以了
        conv5 = tf.nn.relu(bias, name = scope)
        print_activations(conv5)
        parameters += [kernel5, bias5]

        pool5 = tf.nn.max_pool(conv5, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool5')#3*3 W,H  stride = 2
        print_activations(pool5)

        #return pool5, parameters
        # 如果要考虑全连接层，需要再写一些

        # 得到pool5输出的维度
        pool_shape = pool5.get_shape().as_list()

        #计算将矩阵拉直成向量后的长度，这个长度就是矩阵长宽深的乘积
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

        #通过tf.reshape函数将输出变成一个batch的向量
        reshaped = tf.reshape(pool5, [pool_shape[0], -1])#-1用nodes?

        with tf.name_scope('fc1') as scope:
            weights_fc1 = tf.Variable(tf.truncated_normal([nodes, 4096], dtype = tf.float32, stddev = 1e-1), name = 'weights')
            bias_fc1 = tf.Variable(tf.constant(0.0, shape = [4096], dtype = tf.float32), trainable = True, name = 'bias')
            fc1 = tf.nn.relu(tf.matmul(reshaped, weights_fc1) + bias_fc1, name = scope)
            print_activations(fc1)
            parameters += [weights_fc1, bias_fc1]

        with tf.name_scope('fc2') as scope:
            weights_fc2 = tf.Variable(tf.truncated_normal([4096, 4096], dtype = tf.float32, stddev = 1e-1), name = 'weights')
            bias_fc2 = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='bias')
            fc2 = tf.nn.relu(tf.matmul(fc1, weights_fc2) + bias_fc2, name=scope)
            print_activations(fc2)
            parameters += [weights_fc2, bias_fc2]

        with tf.name_scope('fc2') as scope:
            weights_fc3 = tf.Variable(tf.truncated_normal([4096, 1000], dtype = tf.float32, stddev = 1e-1), name = 'weights')
            bias_fc3 = tf.Variable(tf.constant(0.0, shape=[1000], dtype=tf.float32), trainable=True, name='bias')
            fc3 = tf.nn.relu(tf.matmul(fc2, weights_fc3) + bias_fc3, name=scope)
            print_activations(fc3)
            parameters += [weights_fc3, bias_fc3]

        return fc3, parameters

################################################################################
#评估AlexNet每轮计算时间的函数time_tensorflow_run
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

################################################################################
#主函数run_benchmark
#input
#我们并不适用imageNet数据集来训练，只是适用随机图片数据测试前馈和反馈计算耗时。
################################################################################
def run_benchmark():
    with tf.Graph().as_default():#设置默认Graph
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size, image_size, image_size,3], dtype = tf.float32, stddev = 1e-1))#生成随机的图片数据
        fc3, parameters = inference(images)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

    #统计运算时间，传入target的就是pool5，如果有FC层，则是最后一个FC层的输出。然后进行backward。
    #训练时还有一个根据梯度更新参数的过程，不过这个计算量很小，就不统计在评测程序中了。
    time_tensorflow_run(sess, fc3, 'Forward')
    objective = tf.nn.l2_loss(fc3)
    grad = tf.gradients(objective, parameters)#这个函数是tensorflow中用于计算梯度的函数
    time_tensorflow_run(sess, grad, 'Forward-backward')

#最后执行主函数
run_benchmark()