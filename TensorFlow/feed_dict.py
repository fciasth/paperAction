import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

w1 = tf.Variable(tf.random_normal([1,2],stddev=1,seed=1))

x=tf.placeholder(tf.float32,shape=(1,2))
x1=tf.constant([0.7,0.9])

a=x+w1
b=x1+w1
sess=tf.Session()
sess.run(tf.global_variables_initializer())

y_1=sess.run(a,feed_dict={x:[[0.7,0.9]]})
y_2=sess.run(b)
print(y_1)
print(y_2)
sess.close()

