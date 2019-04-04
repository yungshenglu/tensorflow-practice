#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys


# Define hyperparameters
TRAINING_EPOCH = 12
LEARNING_RATE = 0.03
RECORD_STEP = 5
BATCH_SIZE = 64
LAYER_SIZE = 8
ACTIVATION = tf.nn.relu


def fixSeed(seed=1):
	# Reproducible
	np.random.seed(seed)
	tf.set_random_seed(seed)


def updateMeanVar(moving_avg, fc_mean, fc_var):
	moving_avg = moving_avg.apply([fc_mean, fc_var])
	with tf.control_dependencies([moving_avg]):
		return tf.identity(fc_mean), tf.identity(fc_var)


class NNmodel(object):
	def __init__(self, xs, ys, isTrain, batchNorm=False):
		# Set the placeholders
		self.xs = xs
		self.ys = ys
		self.isTrain = isTrain
		self.batchNorm = batchNorm

		# Generate Weights and biases in random
		self.Weights = tf.random_normal_initializer(0.0, 0.1)
		self.biases = tf.constant_initializer(-0.2)

		# Add the input layer
		self.layers = [self.xs]

		# Batch normalization
		if self.batchNorm:
			self.layer_input = [tf.layers.batch_normalization(self.xs, training=self.isTrain)]
		else:
			self.layer_input = [self.xs]
		
		# Add hidden layers
		for _ in range(LAYER_SIZE):
			self.layer_input.append(self.addLayer(self.layer_input[-1], 10, activation=ACTIVATION))

		self.out = tf.layers.dense(self.layer_input[-1], 1, kernel_initializer=self.Weights, bias_initializer=tf.constant_initializer(-0.2))        
		
		# Define the loss function and the optimizer
		self.loss = tf.losses.mean_squared_error(ys, self.out)
		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

	''' Add layer '''
	def addLayer(self, inputs, out_size, activation=None):
		y = tf.layers.dense(inputs, out_size, kernel_initializer=self.Weights, bias_initializer=self.biases)
		self.layers.append(y)

		# Batch normalization
		if self.batchNorm:
			y = tf.layers.batch_normalization(y, momentum=0.4, training=self.isTrain)
		
		# Activation function
		if activation is None:
			return y
		else:
			return activation(y)


''' Plot the histrogram for the inputs of every layer '''
def plotHistogram(axs, layer_input, layer_input_BN, layers, layers_BN, epoch):
	for i, (ax_past, ax_past_BN, ax,  ax_BN) in enumerate(zip(axs[0, :], axs[1, :], axs[2, :], axs[3, :])):
		[a.clear() for a in [ax_past, ax_past_BN, ax, ax_BN]]
		
		# Set the interval
		if i == 0: 
			past_interval = (-7, 10)
			interval = (-7, 10)
		else: 
			past_interval = (-4, 4)
			interval = (-1, 1)

		# Set the title
		ax_past.set_title('L' + str(i))

		# Plot the histogram
		ax_past.hist(layers[i].ravel(), bins=10, range=past_interval, color='#FF9359', alpha=0.5)
		ax_past_BN.hist(layers_BN[i].ravel(), bins=10, range=past_interval, color='#74BCFF', alpha=0.5)
		ax.hist(layer_input[i].ravel(), bins=10, range=interval, color='#FF9359')
		ax_BN.hist(layer_input_BN[i].ravel(), bins=10, range=interval, color='#74BCFF')

		# Set the x-sticks and y-sticks
		for a in [ax_past, ax, ax_past_BN, ax_BN]:
			a.set_xticks(())
			a.set_yticks(())
		ax_past_BN.set_xticks(past_interval)
		ax_BN.set_xticks(interval)

		# Set the label on y axis
		axs[2, 0].set_ylabel('Act')
		axs[3, 0].set_ylabel('BN Act')
	plt.pause(0.01)


def main(arg):
	''' Create data '''
	# Training data
	fixSeed(1)
	x = np.linspace(-7, 10, 2000)[:, np.newaxis]
	np.random.shuffle(x)
	noise = np.random.normal(0, 2, x.shape)
	y = np.square(x) - 5 + noise
	train_data = np.hstack((x, y))

	# Testing data
	xt = np.linspace(-7, 10, 200)[:, np.newaxis]
	noise = np.random.normal(0, 2, xt.shape)
	yt = np.square(xt) -5 + noise

	# Plot the input data
	plt.scatter(x, y, c='#FF9359', s=50, alpha=0.5, label='Train')
	plt.legend(loc='upper left')

	''' Create TensorFlow model '''
	# Define the placeholder for inputs
	xs = tf.placeholder(tf.float32, [None, 1])
	ys = tf.placeholder(tf.float32, [None, 1])
	isTrain = tf.placeholder(tf.bool, None)

	# Set activation function
	global ACTIVATION
	if arg == 'relu':
		ACTIVATION = tf.nn.relu
	elif arg == 'tanh':
		ACTIVATION = tf.nn.tanh

	# Build model with and without BN
	models = [NNmodel(xs=xs, ys=ys, isTrain=isTrain, batchNorm=False), NNmodel(xs=xs, ys=ys, isTrain=isTrain, batchNorm=True)]

	''' Start training '''
	with tf.Session() as sess:
		# Initialize all variables in TensorFlow
		sess.run(tf.global_variables_initializer())

		# Plot the result in figure
		f, axs = plt.subplots(4, LAYER_SIZE + 1, figsize=(10, 5))
		plt.ion()

		# Record all loss with and without batch normalization ###
		losses = [[], []]

		# Training 
		for epoch in range(TRAINING_EPOCH):
			np.random.shuffle(train_data)
			step = 0
			inEpoch = True
			while inEpoch:
				# Get the index of batch (batch_i --> batch_j)
				batch_i, batch_j = (step * BATCH_SIZE) % len(train_data), ((step + 1) * BATCH_SIZE) % len(train_data)
				step += 1
				if batch_j < batch_i:
					batch_j = len(train_data)
					inEpoch = False
				
				# Batch training data
				batch_xs, batch_ys = train_data[batch_i : batch_j, 0 : 1], train_data[batch_i : batch_j, 1 : 2]
				sess.run([models[0].train, models[1].train], feed_dict={
					xs: batch_xs,
					ys: batch_ys,
					isTrain: True
				})

				if step == 1:
					loss, loss_BN, layer_input, layer_input_BN, layers, layers_BN = sess.run([models[0].loss, models[1].loss, models[0].layer_input, models[1].layer_input, models[0].layers, models[1].layers], feed_dict={
						xs: xt, 
						ys: yt,
						isTrain: False
					})
					# Record the loss rate of testing data
					[loss.append(l) for loss, l in zip(losses, [loss, loss_BN])]
					# Plot histogram
					plotHistogram(axs, layer_input, layer_input_BN, layers, layers_BN, epoch)

		plt.ioff()

		# Plot testing loss
		plt.figure(2)
		plt.plot(losses[0], c='#FF9359', lw=3, label='Original')
		plt.plot(losses[1], c='#74BCFF', lw=3, label='Batch Normalization')
		plt.ylabel('test loss')
		plt.ylim((0, 2000))
		plt.legend(loc='best')

		# Prediction
		pred, pred_BN = sess.run([models[0].out, models[1].out], feed_dict={
			xs: xt,
			isTrain: False
		})

		# Plot prediction line
		plt.figure(3)
		plt.plot(xt, pred, c='#FF9359', lw=4, label='Original')
		plt.plot(xt, pred_BN, c='#74BCFF', lw=4, label='Batch Normalization')
		plt.scatter(x[: 200], y[: 200], c='r', s=50, alpha=0.2, label='Train')
		plt.legend(loc='best')
		plt.show()


''' ENTRY POINT '''
if __name__ == "__main__":
	if len(sys.argv) != 2:
		print('[ERROR] No argument')
		print('[INFO] FORMAT: "python3 main.py 1" or "python3 main.py 2"')
		sys.exit()
	else:
		if sys.argv[1] == 'relu':
			print('[INFO] Using ReLU as activation function')
		elif sys.argv[1] == 'tanh':
			print('[INFO] Using tanh as activation function')
		else:
			print('[ERROR] Invalid argument')
			print('[INFO] FORMAT: "python3 main.py 1" or "python3 main.py 2"')
			sys.exit()
		main(sys.argv[1])