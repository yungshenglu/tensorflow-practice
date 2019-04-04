#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D


# Define hyperparameters
TRAINING_EPOCH = 400
LEARNING_RATE = 1
REAL_PARAMS = [1.2, 2.5]
INIT_PARAMS = [[5, 4], [5, 1], [2, 4.5]][2]


''' Select different target function '''
def targetFunction(option, x):
	if option == '1':
		# Use a simple linear function with two parameters
		return lambda w, b: w * x + b
	elif option == '2':
		# Use Tensorflow as a calibrating tool for empirical formula like following
		return lambda w, b: w * x**3 + b * x**2
	elif option == '3':
		# Use the most simplest two parameters and two layers Neural Net, and their local and global minimum
		return lambda w, b: np.sin(b * np.cos(w * x))
	return None


''' Select different training function '''
def trainFunction(option, x):
	if option == '1':
		# Use a simple linear function with two parameters
		return lambda w, b: w * x + b
	elif option == '2':
		# Use Tensorflow as a calibrating tool for empirical formula like following
		return lambda w, b: w * x**3 + b * x**2
	elif option == '3':
		# Use the most simplest two parameters and two layers Neural Net, and their local and global minimum
		return lambda w, b: tf.sin(b * tf.cos(w * x))
	return None


def main(args1, args2):
	''' Create data '''
	# Training data
	x = np.linspace(-1, 1, 200, dtype=np.float32)
	noise = np.random.randn(200) / 10
	y_fun = targetFunction(args1, x)
	y = y_fun(*REAL_PARAMS) + noise

	''' Create TensorFlow model '''
	# Set the learning rate
	global LEARNING_RATE
	LEARNING_RATE = float(args2)

	# Define Weights and biases
	Weights, biases = [tf.Variable(initial_value=val,dtype=tf.float32) for val in INIT_PARAMS]
	
	# Prediction
	y_pred = trainFunction(args1, x)
	pred = y_pred(Weights, biases)

	# Define the loss function and the optimizer
	loss = tf.reduce_mean(tf.square(y - pred))
	train = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

	''' Start training '''
	Weights_list, biases_list, loss_list = [], [], []
	with tf.Session() as sess:
		# Initialize all variables in TensorFlow
		sess.run(tf.global_variables_initializer())

		# Train 400 times
		for epoch in range(TRAINING_EPOCH):
			w, b, l = sess.run([Weights, biases, loss])

			# Record the changes of parameters
			Weights_list.append(w)
			biases_list.append(b)
			loss_list.append(l)

			# Training
			result, _ = sess.run([pred, train])
	
	''' Visualization '''
	print('Weight = ', w, 'bias = ', b)

	# Plot the input data and the training result
	plt.figure(1)
	plt.scatter(x, y, c='#74BCFF', s=50, alpha=0.5, label='Train')
	plt.legend(loc='upper left')
	plt.plot(x, result, 'r-', lw=2)
	plt.savefig('input_%s_%s.png' % (args1, args2))

	# Plot loss rate in 3D figure
	fig = plt.figure(2)
	ax = Axes3D(fig)
	weight3D, bias3D = np.meshgrid(np.linspace(-2, 7, 30), np.linspace(-2, 7, 30))      # Parameter space
	loss3D = np.array([np.mean(np.square(y_fun(w, b) - y)) for w, b in zip(weight3D.flatten(), bias3D.flatten())]).reshape(weight3D.shape)
	ax.plot_surface(weight3D, bias3D, loss3D, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'), alpha=0.5)
	ax.scatter(Weights_list[0], biases_list[0], zs=loss_list[0], s=300, c='r')          # Initial parameter
	ax.set_xlabel('Weight')
	ax.set_ylabel('bias')
	ax.plot(Weights_list, biases_list, zs=loss_list, zdir='z', c='r', lw=3)             # Plot 3D gradient descent
	plt.savefig('%s_%s.png' % (args1, args2))
	plt.show()


''' ENTRY POINT '''
if __name__ == "__main__":
	if len(sys.argv) != 3:
		print('[ERROR] No argument')
		print('[INFO] FORMAT: "python3 main.py 1 [FLOAT]" or "python3 main.py 2 [FLOAT]" or "python3 main.py 3 [FLOAT]"')
		sys.exit()
	else:
		# For option
		if sys.argv[1] == '1':
			print('[INFO] Using linear function as target and training function')
		elif sys.argv[1] == '2':
			print('[INFO] Using non-linear function as target and training function')
		elif sys.argv[1] == '3':
			print('[INFO] Using sin/cos function as target and training function')
		else:
			print('[ERROR] Invalid argument')
			print('[INFO] FORMAT: "python3 main.py 1" or "python3 main.py 2" or "python3 main.py 3"')
			sys.exit()
		
		# For learning rate
		print('[INFO] Using %s learning rate' % sys.argv[2])
		
		main(sys.argv[1], sys.argv[2])