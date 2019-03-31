#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Only show error message
tf.logging.set_verbosity(tf.logging.ERROR)

# Load MNIST datasets from TensorFlow examples
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Define hyperparameters
LEARNING_RATE = 0.01
TRAINING_EPOCH = 20

BATCH_SIZE = 256
INPUT_SIZE = 784        # MNIST data (img: 28*28)
HIDDEN_SIZE_1 = 256     # Number of features in 1st layer
HIDDEN_SIZE_2 = 128     # Number of features in 2nd layer
EXAMPLE_NUM = 10


''' Build the encoder '''
def encoder(xs, Weights, biases):
    # Encode the hidden layer with sigmoid activation
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(xs, Weights['encoder_h1']), biases['encoder_h1']))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, Weights['encoder_h2']), biases['encoder_h2']))
    return layer2


''' Build the decoder '''
def decoder(xs, Weights, biases):
    # Decode the hidden layer with sigmoid activation
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(xs, Weights['decoder_h1']), biases['decoder_h1']))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, Weights['decoder_h2']), biases['decoder_h2']))
    return layer2


def main():
    ''' Create TensorFlow model '''
    # Define the placeholder for inputs
    xs = tf.placeholder(tf.float32, [None, INPUT_SIZE])

    # Define Weights and biases
    Weights = {
        'encoder_h1': tf.Variable(tf.random_normal([INPUT_SIZE, HIDDEN_SIZE_1])),
        'encoder_h2': tf.Variable(tf.random_normal([HIDDEN_SIZE_1, HIDDEN_SIZE_2])),
        'decoder_h1': tf.Variable(tf.random_normal([HIDDEN_SIZE_2, HIDDEN_SIZE_1])),
        'decoder_h2': tf.Variable(tf.random_normal([HIDDEN_SIZE_1, INPUT_SIZE]))
    }
    biases = {
        'encoder_h1': tf.Variable(tf.random_normal([HIDDEN_SIZE_1])),
        'encoder_h2': tf.Variable(tf.random_normal([HIDDEN_SIZE_2])),
        'decoder_h1': tf.Variable(tf.random_normal([HIDDEN_SIZE_1])),
        'decoder_h2': tf.Variable(tf.random_normal([INPUT_SIZE]))
    }

    # Define the encoder (encode) and the decoder (prediction)
    encode = encoder(xs, Weights, biases)
    prediction = decoder(encode, Weights, biases)

    # Define the loss function and the optimizer
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.pow(xs - prediction, 2))
        tf.summary.scalar('loss', loss)
    with tf.name_scope('train'):
        train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    ''' Start training '''
    with tf.Session() as sess:
        # Initialize all variables in TensorFlow
        sess.run(tf.global_variables_initializer())

        # Training (TRAINING_STEP / BATCH_SIZE) times
        for epoch in range(TRAINING_EPOCH):
            # Loop for all batches
            for batch in range(int(mnist.train.num_examples / BATCH_SIZE)):
                batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
                _, loss_rate = sess.run([train, loss], feed_dict={
                    xs: batch_xs
                })
            
            # Show logs per 10 epoch
            if epoch % 1 == 0:
                print('Epoch %3d: Loss = %.9f' % ((epoch + 1), loss_rate))

        # Apply encoder and decoder over test set
        encode_decode = sess.run(prediction, feed_dict={
            xs: mnist.test.images[: EXAMPLE_NUM]
        })

        # Compare original images with their reconstructions
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(EXAMPLE_NUM):
            a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
        plt.show()


''' ENTRY POINT '''
if __name__ == "__main__":
    main()