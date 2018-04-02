import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant(1.0)
b = tf.constant(2.0)
with tf.Session().as_default() as sess:
   print(a.eval())
print(b.eval(session=sess))