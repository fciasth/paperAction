import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# mnist = input_data.read_data_sets('/Users/Nelson/Desktop/Computer/zhihu/denoise_auto_encoder/MNIST_data/')
mnist = input_data.read_data_sets('MNIST_data',validation_size=0,one_hot=False)
img = mnist.train.images[20]
fig = plt.figure()

plt.imshow(img.reshape((28, 28)), cmap='Greys_r')
plt.show()

def get_inputs(real_size,noise_size):
    #真实图像tensor与噪声图像tensor
    real_img = tf.placeholder(tf.float32,[None,real_size],name='real_img')
    noise_img = tf.placeholder(tf.float32,[None,noise_size],name='noise_img')

    return real_img,noise_img

def get_genertor(noise_img,n_units,out_dim,reuse=False,alpha=0.01):
    """"
    noise_img: 生成器的输入
    n_units: 隐层单元个数
    out_dim: 生成器输出tensor的size，这里应该为32*32=784
    alpha: leaky ReLU系数
    """
    with tf.layers.dense(noise_img,n_units):
        #hidden layer
        hidden1 = tf.layers.dense(noise_img,n_units)















