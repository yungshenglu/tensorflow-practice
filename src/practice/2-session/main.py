#!/usr/bin/env python3

import sys
import tensorflow as tf


def main(arg):
	''' Create TensorFlow model '''
	# Define two matrices as constant
	matrix1 = tf.constant([[3, 3]])
	matrix2 = tf.constant([[2],[2]])

	# Product two matrices (same as np.dot(matrix1, matrix2))
	product = tf.matmul(matrix1, matrix2)

	''' Start training '''
	if arg == '1':
		# Method 1 - Without using with
		sess = tf.Session()
		result1 = sess.run(product)
		print(result1)
		sess.close()
	elif arg == '2':
		# Method 2 - Session will be closed in with
		with tf.Session() as sess:
			result2 = sess.run(product)
			print(result2)


''' ENTRY POINT '''
if __name__ == "__main__":
	if len(sys.argv) != 2:
		print('[ERROR] No argument')
		print('[INFO] FORMAT: "python3 main.py 1" or "python3 main.py 2"')
		sys.exit()
	else:
		if sys.argv[1] == '1':
			print('[INFO] Runing TensorFlow session without using with')
		elif sys.argv[1] == '2':
			print('[INFO] Runing TensorFlow session using with')
		else:
			print('[ERROR] Invalid argument')
			print('[INFO] FORMAT: "python3 main.py 1" or "python3 main.py 2"')
			sys.exit()
		main(sys.argv[1])