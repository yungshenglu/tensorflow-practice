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
LEARNING_RATE = 0.01    # 0.01 will be better!
TRAINING_EPOCH = 20

BATCH_SIZE = 256
INPUT_SIZE = 784        # MNIST data (img: 28*28)
HIDDEN_SIZE_1 = 128     # Number of features in 1st layer
HIDDEN_SIZE_2 = 64      # Number of features in 2nd layer
HIDDEN_SIZE_3 = 10      # Number of features in 3rd layer
HIDDEN_SIZE_4 = 2       # Number of features in 4th layer


''' Build the encoder '''
def encoder(x, Weights, biases):
    # Encode the hidden layer with sigmoid activation
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, Weights['encoder_h1']), biases['encoder_h1']))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, Weights['encoder_h2']), biases['encoder_h2']))
    layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, Weights['encoder_h3']), biases['encoder_h3']))
    layer4 = tf.add(tf.matmul(layer3, Weights['encoder_h4']), biases['encoder_h4'])
    return layer4


''' Build the decoder '''
def decoder(x, Weights, biases):
    # Decode the hidden layer with sigmoid activation
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, Weights['decoder_h1']), biases['decoder_h1']))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, Weights['decoder_h2']), biases['decoder_h2']))
    layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, Weights['decoder_h3']), biases['decoder_h3']))
    layer4 = tf.nn.sigmoid(tf.add(tf.matmul(layer3, Weights['decoder_h4']), biases['decoder_h4']))
    return layer4


def main():
    ''' Create TensorFlow model '''
    # Define the placeholder for inputs
    xs = tf.placeholder(tf.float32, [None, INPUT_SIZE])

    # Define Weights and biases
    Weights = {
        'encoder_h1': tf.Variable(tf.random_normal([INPUT_SIZE, HIDDEN_SIZE_1])),
        'encoder_h2': tf.Variable(tf.random_normal([HIDDEN_SIZE_1, HIDDEN_SIZE_2])),
        'encoder_h3': tf.Variable(tf.random_normal([HIDDEN_SIZE_2, HIDDEN_SIZE_3])),
        'encoder_h4': tf.Variable(tf.random_normal([HIDDEN_SIZE_3, HIDDEN_SIZE_4])),
        'decoder_h1': tf.Variable(tf.random_normal([HIDDEN_SIZE_4, HIDDEN_SIZE_3])),
        'decoder_h2': tf.Variable(tf.random_normal([HIDDEN_SIZE_3, HIDDEN_SIZE_2])),
        'decoder_h3': tf.Variable(tf.random_normal([HIDDEN_SIZE_2, HIDDEN_SIZE_1])),
        'decoder_h4': tf.Variable(tf.random_normal([HIDDEN_SIZE_1, INPUT_SIZE]))
    }
    biases = {
        'encoder_h1': tf.Variable(tf.random_normal([HIDDEN_SIZE_1])),
        'encoder_h2': tf.Variable(tf.random_normal([HIDDEN_SIZE_2])),
        'encoder_h3': tf.Variable(tf.random_normal([HIDDEN_SIZE_3])),
        'encoder_h4': tf.Variable(tf.random_normal([HIDDEN_SIZE_4])),
        'decoder_h1': tf.Variable(tf.random_normal([HIDDEN_SIZE_3])),
        'decoder_h2': tf.Variable(tf.random_normal([HIDDEN_SIZE_2])),
        'decoder_h3': tf.Variable(tf.random_normal([HIDDEN_SIZE_1])),
        'decoder_h4': tf.Variable(tf.random_normal([INPUT_SIZE]))
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
        total_batch = int(mnist.train.num_examples / BATCH_SIZE)
        for epoch in range(TRAINING_EPOCH):
            # Loop for all batches
            for batch in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
                _, loss_rate = sess.run([train, loss], feed_dict={
                    xs: batch_xs
                })
            
            # Show logs per epoch
            if epoch % 1 == 0:
                print('Epoch %2d: Loss = %.9f' % ((epoch + 1), loss_rate))

        # Show the result after encoding
        encode_result = sess.run(encode, feed_dict={
            xs: mnist.test.images
        })
        plt.scatter(encode_result[:, 0], encode_result[:, 1], c=np.argmax(mnist.test.labels, 1)[:])
        plt.colorbar()
        plt.show()


''' ENTRY POINT '''
if __name__ == "__main__":
    main()