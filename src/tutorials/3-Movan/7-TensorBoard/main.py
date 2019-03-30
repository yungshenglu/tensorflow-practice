#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


''' Add layer '''
def addLayer(inputs, in_size, out_size, name, activation=None):
    # Define Weights and biases and add the name for TensorBoard
    layer_name = 'Layer%s' % name
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        
        # Define the training function and add the name for TensorBoard
        with tf.name_scope('y'):
            y = tf.add(tf.matmul(inputs, Weights), biases)
        
        # Activation function
        if activation is None:
            outputs = y
        else:
            outputs = activation(y)
        tf.summary.histogram(layer_name + '/outputs', outputs)
        
        return outputs


def main():
    ''' Create data '''
    x = np.linspace(-1, 1, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x.shape)
    y = np.square(x) - 0.5 + noise

    ''' Create TensorFlow model '''
     # Add the name of the placeholder for TensorBoard
    with tf.name_scope('inputs'):
        xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
        ys = tf.placeholder(tf.float32, [None, 1], name='y_input')
    
    # Add layer
    layer1 = addLayer(xs, 1, 10, name=1, activation=tf.nn.relu)
    prediction = addLayer(layer1, 10, 1, name=2, activation=None)

    # Define the loss function and the optimizer
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
        tf.summary.scalar('loss', loss)
    with tf.name_scope('train'):
        train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    ''' Start training '''
    with tf.Session() as sess:
        # Merge all summary and write into a file
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./logs/', sess.graph)

        # Initialize all variables in TensorFlow
        sess.run(tf.global_variables_initializer())

        # Train 1000 times
        for step in range(1000):
            sess.run(train, feed_dict={
                xs: x,
                ys: y
            })
            # Write into summary for every 50 times
            if step % 50 == 0:
                result = sess.run(merged, feed_dict={
                    xs: x,
                    ys: y
                })
                writer.add_summary(result, step)


''' ENTRY POINT '''
if __name__ == "__main__":
    main()