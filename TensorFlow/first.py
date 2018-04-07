import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
matrix1 = tf.constant([[3.,3.]])

matrix2 = tf.constant([[2.],[2.]])
#构造一个线性模型

product = tf.matmul(matrix1,matrix2)

# sess = tf.Session()
#
# result = sess.run(product)
# print(result)
#
# sess.close()

with tf.Session() as sess:
    with tf.device("/gpu:1"):
        result = sess.run([product])
        print(result)