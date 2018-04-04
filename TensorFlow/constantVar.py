import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant(2,tf.int16)
b = tf.constant(4,tf.float32)
c = tf.constant(8,tf.float32)

d = tf.Variable(2,tf.int16)
e = tf.Variable(4,tf.float32)
f = tf.Variable(8,tf.float32)

g = tf.constant(np.zeros(shape=(2,2),dtype=np.float32))

h=tf.zeros([11],tf.int16)
i=tf.ones([2,2,],tf.float32)
j=tf.zeros([1000,4,3],tf.float64)

k=tf.Variable(tf.zeros([2,2],tf.float32))
l=tf.Variable(tf.zeros([5,6,5],tf.float32))

weight = tf.Variable(tf.truncated_normal([256*256,10]))
biases = tf.Variable(tf.zeros([10]))
print(weight.get_shape().as_list())
print(biases.get_shape().as_list())