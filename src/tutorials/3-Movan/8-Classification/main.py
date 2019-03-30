#!/usr/bin/env python3

import tensorflow as tf


# Only show error message
tf.logging.set_verbosity(tf.logging.ERROR)

# Load MNIST datasets from TensorFlow examples
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


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


''' Compute the accuracy '''
def computeAccruracy(sess, prediction, xs, ys, v_xs, v_ys):
    # Prediction
    y_pre = sess.run(prediction, feed_dict={
        xs: v_xs
    })

    # Check whether the prediction is correct
    correct = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))

    # Comput the accuracy
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    return sess.run(accuracy, feed_dict={
        xs: v_xs,
        ys: v_ys
    })


def main():
    ''' Create TensorFlow model '''
    # Define the placeholder for inputs
    xs = tf.placeholder(tf.float32, [None, 784])
    ys = tf.placeholder(tf.float32, [None, 10])

    # Add layer
    prediction = addLayer(xs, 784, 10, activation=tf.nn.softmax)

    # Define the loss function (cross entropy) and the optimizer
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
    train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    ''' Start training '''
    with tf.Session() as sess:
        # Initialize all variables in TensorFlow
        sess.run(tf.global_variables_initializer())

        # Train 1000 times
        for step in range(1000):
            # Batch the MNIST datasets for every 100
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train, feed_dict={
                xs: batch_xs,
                ys: batch_ys
            })

            # Print the accuracy for every 50 times
            if step % 50 == 0:
                print('Step %3d' % step, computeAccruracy(sess, prediction, xs, ys, mnist.test.images, mnist.test.labels))

        


''' ENTRY POINT '''
if __name__ == "__main__":
    main()