# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 17:29:00 2018

@author: douyh
"""

'''
reference:《TensorFlow实战》黄文坚等
step:
1.定义算法公式，也就是神经网络forward时的计算
2.定义loss，选定优化器，并制定优化器优化loss
3.迭代地对数据进行训练
4.在测试集上对准确率进行评测
'''
import tensorflow as tf
'''
导入数据，但是好像直接这样这样用会报错，应该是数据包加载的过程中出现了问题，需要首先在网上下载下来数据集的压缩包.
You can download the datasets on the website: http://yann.lecun.com/exdb/mnist/ or
download them from my URL: http://pan.baidu.com/s/1mh0ICBI ( password: dlut ).
Four files are available: 
 - train-images-idx3-ubyte.gz : training set images (9912422 bytes) 
 - train-labels-idx1-ubyte.gz : training set labels (28881 bytes) 
 - t10k-images-idx3-ubyte.gz  : test set images (1648877 bytes) 
 - t10k-labels-idx1-ubyte.gz  : test set labels (4542 bytes)
'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
'''
print test; print size of images
'''
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.images.shape)
'''
创建新的InteractiveSession,使用这个命令会将这个session注册为默认session，之后的运算也都默认跑在这个session里
不同的session质检的数据和运算应该是相互独立的
接下来创建一个Placeholder，即输入数据的地方。
Placeholder第一个参数是数据类型，第二个参数[None, 784]代表Tensor的shape，也就是数据的尺寸。
这里的None代表不限条数的输入，784代表每条输入是一个784维的向量。
'''
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])

'''
给softmax regression模型中的weights和bias创建Variable对象
Variable是用来存储模型参数的
'''
W = tf.Variable(tf.zeros([784, 10]))#初始化为0
b = tf.Variable(tf.zeros([10]))

'''
实现softmax regression 算法
y = softmax(Wx + b)
tensorflow定义了softmax regression，语法和直接写公式很像，而且将forward和backward的内容都自动实现
只要接下来定义好loss，训练时将会自动求导并进行梯度下降，完成对softmax模型参数的自动学习
'''
y = tf.nn.softmax(tf.matmul(x, W) + b) #tf.nn包含了大量的神经网络的组件

'''
定义一个loss funtion
对于多分类问题，通常使用cross-entropy作为loss function
先定义一个placeholder输入真是的label，用来计算交叉熵。tf.reduce_sum是求和，tf.reduce_mean是对每个batch数据求均值
'''
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))

'''
分类方法softmax regression 和 损失函数 cross-entropy都已经确定了，接下来需要定义一个优化算法
采用常见的随机梯度下降SGD(stochastic gradient descent),自动求导，并且根据BP算法进行训练，在每一轮迭代时更新参数
直接调用函数，设置学习速率为0.5(learning rate/ step length),优化目标设定为cross-entropy,得到进行训练的操作train_step.
'''
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

'''
使用TF的全局参数初始化器，并直接run
'''
tf.global_variables_initializer().run()

'''
最后一步，迭代执行训练操作train_step
每次都从训练集中抽取出100条样本构成mini_batch，并feed给placeholder,然后调用train_step对这些样本进行训练
'''
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs, y_:batch_ys})

'''
现在训练已经完成，接下来可以对模型的准确率进行验证
tf.argmax是从一个tensor中找到最大值的序号，tf.equal是判断是否相同
'''
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

'''
统计全部样本预测的精确度
这里需要用tf.cast将之前correct_prediction输出的bool值转换为float32，再求平均
'''
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))#

'''
将测试数据的特征和label输入评测流程accuracy,计算模型再测试机上的准确率并将结果打印出来
'''
#print(accuracy.eval({x: mnist.test.images, y_:mnist.test.labels}))#此语句与下一行的语句等价
print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
