#!/usr/bin/env python3

import tensorflow as tf


# Only show error message
tf.logging.set_verbosity(tf.logging.ERROR)

# Load MNIST datasets from TensorFlow examples
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


''' Create RNN '''
def RNN(x, weights, biases):
    # Create a hidden layer for input to cell


    # Create cell


    # Create a hidden layer for output to the final results

    results = None
    return results


def main():
    ''' Define hyperparameters '''
    learning_rate = 0.001
    training_step = 100000
    batch_size = 128

    n_inputs = 28           # Width of each MNIST data (img: 28*28)
    n_steps = 28            # Height of each MNIST data (img: 28*28)
    n_hidden_units = 128    # Neurons in hidden layer
    n_classes = 10          # Classes of MNIST data (0-9 digits)

    ''' Create TensorFlow model '''
    # Define the placeholder for inputs
    xs = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    ys = tf.placeholder(tf.float32, [None, n_classes])

    # Define Weights and biases for input and output respectively
    Weights = {
        'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),    # size = 28*128
        'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))   # size = 128*10
    }
    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units])),        # size = 128
        'out': tf.Variable(tf.constant(0.1, shape=[n_classes]))             # size = 10
    }

    # Add RNN layers
    prediction = RNN(xs, Weights, biases)

    # Define the loss function (cross entropy) and the optimizer
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, ys))
        tf.summary.scalar('loss', loss)
    with tf.name_scope('train'):
        train = tf.train.AdamOptimizer(learning_rate).minimize(loss)



''' ENTRY POINT '''
if __name__ == "__main__":
    main()