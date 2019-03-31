#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Define hyperparameters
LEARNING_RATE = 0.006
TRAINING_STEP = 200

BATCH_SIZE = 50
BATCH_START = 0
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 10
TIME_STEP = 20 


class LSTMRNN(object):
    def __init__(self, batch_size, input_size, output_size, cell_size, time_step, learning_rate):
        # Set hyperparameters
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.time_step = time_step
        self.learning_rate = learning_rate

        ''' Create TensorFlow model '''
        # Define the placeholder for inputs 
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, self.time_step, self.input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, self.time_step, self.output_size], name='ys')
        
        # Create a hidden layer for input to cell
        with tf.variable_scope('hidden_input'):
            self.addInputLayer()
        
        # Create LSTM cell
        with tf.variable_scope('cell'):
            self.addCell()
        
        # Create a hidden layer for output to the final results
        with tf.variable_scope('hidden_output'):
            self.addOutputLayer()
        
        # Define the loss function
        with tf.name_scope('loss'):
            self.computeLoss()
        
        # Define the optimizer
        with tf.name_scope('train'):
            self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)


    # Create a hidden layer for input to cell
    def addInputLayer(self):
        # xs_in (50 batch * steps, 1 inputs)
        xs_in = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')

        # Define Weights and biases for each cell
        Weights_in = self.weightVar([self.input_size, self.cell_size])
        biases_in = self.biasVar([self.cell_size])

        # ys_in (50 batch * steps, 10 cell)
        with tf.name_scope('ys_in'):
            ys_in = tf.add(tf.matmul(xs_in, Weights_in), biases_in)

        # ys_in (50 batch, steps, 10 cell)
        self.ys_in = tf.reshape(ys_in, [-1, self.time_step, self.cell_size], name='2_3D')


    # Create LSTM cell
    def addCell(self):
        cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        
        # LSTM cell is divided into two parts (c_state, m_state)
        with tf.name_scope('initial_state'):
            self.cell_init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
        
        # Recurrent cell
        # time_major : the time is in the 1st dimension or not
        self.cell_out, self.cell_final_state = tf.nn.dynamic_rnn(cell, self.ys_in, initial_state=self.cell_init_state, time_major=False)
    

    # Create a hidden layer for output to the final results
    def addOutputLayer(self):
        # xs_out (50 batch * steps, 10 cell)
        xs_out = tf.reshape(self.cell_out, [-1, self.cell_size], name='2_2D')

        # Define Weights and biases for each cell
        Weights_out = self.weightVar([self.cell_size, self.output_size])
        biases_out = self.biasVar([self.output_size])

         # ys_out (50 batch * steps, 1 outputs)
        with tf.name_scope('ys_out'):
            self.pred = tf.add(tf.matmul(xs_out, Weights_out), biases_out)
    

    def computeLoss(self):
        # Compute the loss rate for each step
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_prediction')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.time_step], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.meanSquareError,
            name='losses'
        )

        # Compute the average loss
        with tf.name_scope('average_loss'):
            self.loss = tf.div(tf.reduce_sum(losses, name='sum_loss'), self.batch_size, name='average_loss')
            tf.summary.scalar('loss', self.loss)


    # Define the variable of weight
    def weightVar(self, shape, name='Weights'):
        init = tf.random_normal_initializer(mean=0.0, stddev=1.0)
        return tf.get_variable(shape=shape, initializer=init, name=name)
    

    # Define the variable of bias
    def biasVar(self, shape, name='biases'):
        init = tf.constant_initializer(0.1)
        return tf.get_variable(shape=shape, initializer=init, name=name)

    
    # Define the mean-square error function
    @staticmethod
    def meanSquareError(labels, logits):
        return tf.square(tf.subtract(labels, logits))

def getBatch():
    global BATCH_START, TIME_STEP

    # xs_reshape (50 batch, 20 steps)
    xs_reshape = np.arange(BATCH_START, BATCH_START + TIME_STEP * BATCH_SIZE).reshape([BATCH_SIZE, TIME_STEP]) / (10 * np.pi)
    seq = np.sin(xs_reshape)
    res = np.cos(xs_reshape)
    BATCH_START += TIME_STEP

    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs_reshape]



def main():
    ''' Create TensorFlow model '''
    model = LSTMRNN(BATCH_SIZE, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, TIME_STEP, LEARNING_RATE)

    ''' Start training '''
    with tf.Session() as sess:
        # Merge all summary and write into a file
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./logs/', sess.graph)

        # Initialize all variables in TensorFlow
        sess.run(tf.global_variables_initializer())

        # Plot the result in figure
        plt.ion()
        plt.show()

        # Train 1000 times
        for step in range(200):
            seq, res, xs = getBatch()
            if step == 0:
                # No need to define the initial state!
                feed_dict = {
                    model.xs: seq,
                    model.ys: res
                }
            else:
                feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    model.cell_init_state: state
                }
            
            # Training
            _, loss, state, pred = sess.run(
                [model.train, model.loss, model.cell_final_state, model.pred],
                feed_dict=feed_dict
            )

            # Plotting
            plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], pred.flatten()[: TIME_STEP], 'b--')
            plt.ylim([-1.2, 1.2])
            plt.draw()
            if step % 5 == 0:
                plt.savefig('%s.png' % step)
            plt.pause(0.3)

            # Print the accuracy for every 20 times
            if step % 20 == 0:
                print('Step %3d: Loss = %f' % (step, round(loss, 4)))
                result = sess.run(merged, feed_dict=feed_dict)
                writer.add_summary(result, step)


''' ENTRY POINT '''
if __name__ == "__main__":
    main()