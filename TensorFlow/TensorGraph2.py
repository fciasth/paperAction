import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant([1,2],name="a")
b = tf.constant([3,4],name="b")

result = a + b
# print(result)
sess = tf.Session()
print(sess.run(result))
sess.close()

with tf.Session() as sess:
    a=tf.constant([1,2,3,4])
    b=tf.constant([1,2,3,4])
    result=a+b
    print(sess.run(result))