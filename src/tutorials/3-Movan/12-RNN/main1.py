#!/usr/bin/env python3

import tensorflow as tf


# Only show error message
tf.logging.set_verbosity(tf.logging.ERROR)

# Load MNIST datasets from TensorFlow examples
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Define hyperparameters
LEARNING_RATE = 0.001
TRAINING_STEP = 100000

BATCH_SIZE = 128
INPUT_SIZE = 28         # Width of each MNIST data (img: 28*28)
OUTPUT_SIZE = 10        # Classes of MNIST data (0-9 digits)
CELL_SIZE = 128         # Neurons in hidden layer    
TIME_STEP = 28          # Height of each MNIST data (img: 28*28)  


''' Compute the accuracy '''
def computeAccruracy(sess, prediction, xs, ys, v_xs, v_ys):
    # Prediction
    y_pred = sess.run(prediction, feed_dict={
        xs: v_xs,
    })

    # Check whether the prediction is correct
    correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(v_ys, 1))

    # Comput the accuracy
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    return sess.run(accuracy, feed_dict={
        xs: v_xs,
        ys: v_ys
    })


''' Create RNN '''
def RNN(xs, Weights, biases):
    ''' Create a hidden layer for input to cell '''
    # xs (128 batch, 28 steps, 28 inputs) --> (128 batch * 28 steps, 28 inputs)
    xs = tf.reshape(xs, [-1, INPUT_SIZE])

    # x_in (128 batch * 28 steps, 28 hidden_units)
    xs_in = tf.add(tf.matmul(xs, Weights['in']), biases['in'])

    # x_in (128 batch, 28 steps, 28 hidden_units)
    xs_in = tf.reshape(xs_in, [-1, TIME_STEP, CELL_SIZE])

    ''' Create LSTM cell '''
    cell = tf.nn.rnn_cell.BasicLSTMCell(CELL_SIZE)

    # LSTM cell is divided into two parts (c_state, m_state)
    init_state = cell.zero_state(BATCH_SIZE, dtype=tf.float32)

    # Recurrent cell
    # time_major : the time is in the 1st dimension or not
    outputs, final_state = tf.nn.dynamic_rnn(cell, xs_in, initial_state=init_state, time_major=False)


    ''' Create a hidden layer for output to the final results '''
    # Unstack to the list [(batch, outputs) ...] * steps 
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    return tf.add(tf.matmul(outputs[-1], Weights['out']), biases['out'])


def main():
    ''' Create TensorFlow model '''
    # Define the placeholder for inputs
    xs = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])
    ys = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])

    # Define Weights and biases for each cell
    Weights = {
        'in': tf.Variable(tf.random_normal([INPUT_SIZE, CELL_SIZE])),    # size = 28*128
        'out': tf.Variable(tf.random_normal([CELL_SIZE, OUTPUT_SIZE]))   # size = 128*10
    }
    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[CELL_SIZE])),        # size = 128*1
        'out': tf.Variable(tf.constant(0.1, shape=[OUTPUT_SIZE]))             # size = 10*1
    }

    # Define LSTM RNN
    prediction = RNN(xs, Weights, biases)

    # Define the loss function (cross entropy) and the optimizer
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))
        tf.summary.scalar('loss', loss)
    with tf.name_scope('train'):
        train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    ''' Start training '''
    with tf.Session() as sess:
        # Initialize all variables in TensorFlow
        sess.run(tf.global_variables_initializer())

        # Training (TRAINING_STEP / BATCH_SIZE) times
        step = 0
        while step * BATCH_SIZE < TRAINING_STEP:
            batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
            batch_xs = batch_xs.reshape([BATCH_SIZE, TIME_STEP, INPUT_SIZE])
            sess.run(train, feed_dict={
                xs: batch_xs,
                ys: batch_ys
            })

            # Print the accuracy for every 50 times
            if step % 50 == 0:
                test_data = mnist.test.images[: BATCH_SIZE].reshape([-1, TIME_STEP, INPUT_SIZE])
                test_label = mnist.test.labels[: BATCH_SIZE]
                print('Step %3d:' % step, computeAccruracy(sess, prediction, xs, ys, test_data, test_label))
            step += 1


''' ENTRY POINT '''
if __name__ == "__main__":
    main()