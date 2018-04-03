import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
graph = tf.Graph()
with graph.as_default():
    a = tf.Variable(8,tf.float32)
    b = tf.Variable(tf.zeros([2,2],tf.float32))

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    #print(f)
    print(session.run(a))
    print(session.run(b))