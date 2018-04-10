import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
#提供了一份python源代码用于自动下载和安装这个数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sess = tf.InteractiveSession()
x = tf.placeholder("float",shape=[None,784])
y_ = tf.placeholder("float",shape=[None,10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
sess.run(tf.global_variables_initializer())

y = tf.nn.softmax(tf.matmul(W,x)+b)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


for i in range(50):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print (accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
