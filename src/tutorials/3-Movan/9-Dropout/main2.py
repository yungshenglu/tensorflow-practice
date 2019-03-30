#!/usr/bin/env python3

import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def addLayer(inputs, in_size, out_size, keep_prob, name, activation=None):
    # Define Weights and biases and add the name for TensorBoard
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(name + '/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(name + '/biases', biases)
        
        # Define the training function (with dropout) and add the name for TensorBoard
        with tf.name_scope('y'):
            y = tf.add(tf.matmul(inputs, Weights), biases)
            # Dropout
            y = tf.nn.dropout(y, keep_prob)
        
        # Activation function
        if activation is None:
            outputs = y
        else:
            outputs = activation(y)
        tf.summary.histogram(name + '/outputs', outputs)
        
        return outputs


def main():
    ''' Load data '''
    digits = load_digits()
    x = digits.data
    y = LabelBinarizer().fit_transform(digits.target)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    ''' Create TensorFlow model '''
    # Define the placeholder for inputs and keep_prob (for dropout)
    xs = tf.placeholder(tf.float32, [None, 64])
    ys = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    # Add layer
    layer1 = addLayer(xs, 64, 50, keep_prob=keep_prob, name='layer1', activation=tf.nn.tanh)
    prediction = addLayer(layer1, 50, 10, keep_prob=keep_prob, name='layer2', activation=tf.nn.softmax)

    # Define the loss function (cross entropy) and the optimizer
    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
        tf.summary.scalar('loss', cross_entropy)
    with tf.name_scope('train'):
        train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    ''' Start training '''
    with tf.Session() as sess:
        # Merge all summary and write into a file
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
        test_writer = tf.summary.FileWriter('./logs/test', sess.graph)

        # Initialize all variables in TensorFlow
        sess.run(tf.global_variables_initializer())

        # Train 1000 times
        for step in range(1000):
            # Let the dropout rate be 0.5 (= 1 - keep_prob)
            sess.run(train, feed_dict={
                xs: x_train,
                ys: y_train,
                keep_prob: 0.5
            })

            # Write into summary for every 50 times
            if step % 50 == 0:
                train_result = sess.run(merged, feed_dict={
                    xs: x_train,
                    ys: y_train,
                    keep_prob: 1
                })
                test_result = sess.run(merged, feed_dict={
                    xs: x_test,
                    ys: y_test,
                    keep_prob: 1
                })
                train_writer.add_summary(train_result, step)
                test_writer.add_summary(test_result, step)


''' ENTRY POINT '''
if __name__ == "__main__":
    main()