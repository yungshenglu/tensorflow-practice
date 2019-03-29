#!/usr/bin/env python3

import tensorflow as tf
import numpy as np


def main():
	''' Create data '''
	x = np.random.rand(100).astype(np.float32)
	y = x * 0.7 + 0.5

	''' Create TensorFlow model '''
	# Define Weights and biases
	Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
	biases = tf.Variable(tf.zeros([1]))

	# Define training function
	y_train = Weights * x + biases

	# Define loss function and optimizer
	loss = tf.reduce_mean(tf.square(y_train - y))
	optimizer = tf.train.GradientDescentOptimizer(0.5)
	train = optimizer.minimize(loss)

	# Initialize all variables in TensorFlow
	init = tf.initialize_all_variables()

	''' Start training '''
	sess = tf.Session()
	sess.run(init)

	# Print the Weights and biases for every 20 steps
	for step in range(201):
	    sess.run(train)
	    if step % 20 == 0:
		    print(step, sess.run(Weights), sess.run(biases))


''' ENTRY POINT '''
if __name__ == "__main__":
    main()