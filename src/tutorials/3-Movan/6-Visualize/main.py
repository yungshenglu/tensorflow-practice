#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


''' Add layer '''
def addLayer(inputs, in_size, out_size, activation=None):
    # Define Weights and biases
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    
    # Define the training function
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

    ''' Create TensorFlow model '''
    # Define the placeholder for inputs
    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])

    # Add layer
    layer1 = addLayer(xs, 1, 10, activation=tf.nn.relu)
    prediction = addLayer(layer1, 10, 1, activation=None)

    # Define the loss function and the optimizer
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    ''' Start training '''
    with tf.Session() as sess:
        # Initialize all variables in TensorFlow
        sess.run(tf.global_variables_initializer())

        # Plot the result in figure
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(x, y)
        plt.ion()
        plt.show()

        # Print the Weights and biases for every 50 steps
        for step in range(1000):
            sess.run(train, feed_dict={
                xs: x,
                ys: y
            })
            if step % 50 == 0:
                #print(step, sess.run(loss, feed_dict={
                #    xs: x,
                #    ys: y
                #}))
                # Remove the previous line and plot the value of prediction in curve line
                try:
                    ax.lines.remove(lines[0])
                except Exception:
                    pass
                prediction_value = sess.run(prediction, feed_dict={
                    xs: x
                })
                lines = ax.plot(x, prediction_value, 'r-', lw=3)
                plt.pause(0.1)


''' ENTRY POINT '''
if __name__ == "__main__":
    main()