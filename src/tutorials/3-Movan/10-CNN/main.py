#!/usr/bin/env python3

import tensorflow as tf


# Only show error message
tf.logging.set_verbosity(tf.logging.ERROR)

# Load MNIST datasets from TensorFlow examples
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def computeAccruracy(sess, prediction, keep_prob, xs, ys, v_xs, v_ys):
    # Prediction
    y_pre = sess.run(prediction, feed_dict={
        xs: v_xs,
        keep_prob: 1
    })

    # Check whether the prediction is correct
    correct = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))

    # Comput the accuracy
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    return sess.run(accuracy, feed_dict={
        xs: v_xs,
        ys: v_ys,
        keep_prob: 1
    })

''' Define the variable of weight '''
def weightVar(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


''' Define the variable of bias '''
def biasVar(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


''' Create 2D convolution layer '''
def conv2d(x, weights):
    # Notes: stride = [1, x_movement, y_movement, 1] (stride[0] = stride[3] = 1)
    return tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')


''' Create max-pooling layer '''
def maxPool2x2(x):
    # Notes: stride = [1, x_movement, y_movement, 1] (stride[0] = stride[3] = 1)
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def main():
    ''' Create TensorFlow model '''
    # Define the placeholder for inputs and keep_prob (for dropout)
    xs = tf.placeholder(tf.float32, [None, 784]) / 255
    ys = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    # Reshape input image
    x_image = tf.reshape(xs, [-1, 28, 28, 1])

    # Add conv1 layer
    w_conv1 = weightVar([5, 5, 1, 32])                                  # patch = 5*5, in_size = 1, out_size = 32
    b_conv1 = biasVar([32])
    h_conv1 = tf.nn.relu(tf.add(conv2d(x_image, w_conv1),  b_conv1))    # out_size = 28*28*32
    h_pool1 = maxPool2x2(h_conv1)                                       # out_size = 14*14*32

    # Add conv2 layer
    w_conv2 = weightVar([5, 5, 32, 64])                                 # patch = 5*5, in_size = 32, out_size = 64
    b_conv2 = biasVar([64])
    h_conv2 = tf.nn.relu(tf.add(conv2d(h_pool1, w_conv2),  b_conv2))    # out_size = 14*14*64
    h_pool2 = maxPool2x2(h_conv2)                                       # out_size = 7*7*64

    # Add fc1 layer
    w_fc1 = weightVar([7*7*64, 1024])
    b_fc1 = biasVar([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])                    # [n_sample, 7, 7, 64] >> [n_sample, 7*7*64]
    h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_pool2_flat, w_fc1), b_fc1))
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Add fc2 alyer
    w_fc2 = weightVar([1024, 10])
    b_fc2 = biasVar([10])
    prediction = tf.nn.softmax(tf.add(tf.matmul(h_fc1_drop, w_fc2), b_fc2))

    # Define the loss function (cross entropy) and the optimizer
    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
        tf.summary.scalar('loss', cross_entropy)
    with tf.name_scope('train'):
        train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
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
                ys: batch_ys,
                keep_prob: 0.5
            })

            # Print the accuracy for every 50 times
            if step % 50 == 0:
                print('Step %3d' % step, computeAccruracy(sess, prediction, keep_prob, xs, ys, mnist.test.images[: 1000], mnist.test.labels[: 1000]))


''' ENTRY POINT '''
if __name__ == "__main__":
    main()