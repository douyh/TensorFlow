# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 16:42:48 2018

@author: douyh
"""

# 进入一个交互式 TensorFlow 会话.
import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# 使用初始化器 initializer op 的 run() 方法初始化 'x' 
x.initializer.run()

# 增加一个减法 sub op, 从 'x' 减去 'a'. 运行减法 op, 输出结果 
sub = tf.subtract(x, a)#tf.sub没有了,现在有tf.subtract
print (sub.eval())
# ==> [-2. -1.]