#!/usr/bin/env python3

import tensorflow as tf


def main():
	''' Create data '''
	# Define variables
	state = tf.Variable(0, name='counter')
	one = tf.constant(1)

	# Add one into new_state and update to state
	new_state = tf.add(state, one)
	update = tf.assign(state, new_state)

	# Initialize all variables in TensorFlow
	init = tf.initialize_all_variables()

	''' Start training '''
	with tf.Session() as sess:
		sess.run(init)
		for _ in range(10):
			sess.run(update)
			print(sess.run(state))


''' ENTRY POINT '''
if __name__ == "__main__":
    main()