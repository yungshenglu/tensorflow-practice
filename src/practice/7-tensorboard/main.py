#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def addLayer(inputs, in_size, out_size, activation=None):
    # Define Weights and biases and add the name for TensorBoard
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        
        # Define training function and add the name for TensorBoard
        with tf.name_scope('y'):
            y = tf.add(tf.matmul(inputs, Weights), biases)
        
        # Activation function
        if activation is None:
            outputs = y
        else:
            outputs = activation(y)
        
        return outputs


def main():
    ''' Create data '''
    x = np.linspace(-1, 1, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x.shape)
    y = np.square(x) - 0.5 + noise

    # Add name for TensorBoard
    with tf.name_scope('inputs'):
        xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
        ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

    ''' Create TensorFlow model '''
    layer1 = addLayer(xs, 1, 10, activation=tf.nn.relu)
    prediction = addLayer(layer1, 10, 1, activation=None)

    # Define loss function and optimizer
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    with tf.name_scope('train'):
        train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    ''' Start training '''
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./logs/', sess.graph)
        # Initialize all variables in TensorFlow
        sess.run(tf.global_variables_initializer())


''' ENTRY POINT '''
if __name__ == "__main__":
    main()