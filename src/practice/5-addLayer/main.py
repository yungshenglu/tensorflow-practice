#!/usr/bin/env python3

import tensorflow as tf
import numpy as np


def addLayer(inputs, in_size, out_size, activation=None):
    # Define Weights and biases
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    
    # Define training function
    y = tf.matmul(inputs, Weights) + biases
    
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

    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])

    ''' Create TensorFlow model '''
    layer1 = addLayer(xs, 1, 10, activation=tf.nn.relu)
    prediction = addLayer(layer1, 10, 1, activation=None)

    # Define loss function and optimizer
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    # Initialize all variables in TensorFlow
    init = tf.initialize_all_variables()

    ''' Start training '''
    with tf.Session() as sess:
        sess.run(init)
        for step in range(1000):
            sess.run(train, feed_dict={
                xs: x,
                ys: y
            })
            if step % 50 == 0:
                print(step, sess.run(loss, feed_dict={
                    xs: x,
                    ys: y
                }))


''' ENTRY POINT '''
if __name__ == "__main__":
    main()