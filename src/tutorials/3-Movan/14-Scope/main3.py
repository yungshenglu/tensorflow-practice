#!/usr/bin/env python3

import tensorflow as tf


''' Define training configurations '''
class TrainConfig:
    TRAIN_EPOCH = 20
    LEARNING_RATE = 0.01
    BATCH_SIZE = 20
    INPUT_SIZE = 10
    OUTPUT_SIZE = 2
    CELL_SIZE = 11


''' Define testing configurations '''
class TestConfig(TrainConfig):
    TRAIN_EPOCH = 1


''' Define RNN model '''
class RNN(object):
    def __init__(self, config):
        # Set hyperparameters
        self.batch_size = config.BATCH_SIZE
        self.train_epoch = config.TRAIN_EPOCH
        self.input_size = config.INPUT_SIZE
        self.output_size = config.OUTPUT_SIZE
        self.cell_size = config.CELL_SIZE
        self.learning_rate = config.LEARNING_RATE

        # Build RNN model
        self.buildRNN()

    def buildRNN(self):
        # Define the placeholder for inputs 
        with tf.variable_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [self.batch_size, self.train_epoch, self.input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [self.batch_size, self.train_epoch, self.output_size], name='ys')
        
        with tf.name_scope('RNN'):
            # Create a hidden layer for input to cell
            with tf.variable_scope('input_layer'):
                # xs_in (20 batch * 20 steps, 1 inputs)
                xs_in = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')

                # Define Weights and biases for each cell
                Weights_in = self.weightVar([self.input_size, self.cell_size])
                print(Weights_in.name)
                biases_in = self.biasVar([self.cell_size])

                # ys_in (20 batch * 20 steps, 11 cell)
                with tf.name_scope('Wx_plus_b'):
                    ys_in = tf.matmul(xs_in, Weights_in) + biases_in
                ys_in = tf.reshape(ys_in, [-1, self.train_epoch, self.cell_size], name='2_3D')

            # Create LSTM cell
            with tf.variable_scope('cell'):
                cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size)

                # LSTM cell is divided into two parts (c_state, m_state)
                with tf.name_scope('initial_state'):
                    self.cell_init_state = cell.zero_state(self.batch_size, dtype=tf.float32)

                self.cell_outputs = []
                cell_state = self.cell_init_state
                for epoch in range(self.train_epoch):
                    if epoch > 0: 
                        tf.get_variable_scope().reuse_variables()
                    cell_output, cell_state = cell(ys_in[:, epoch, :], cell_state)
                    self.cell_outputs.append(cell_output)
                self.cell_final_state = cell_state

            # Create a hidden layer for output to the final results
            with tf.variable_scope('output_layer'):
                # xs_out (20 batch * 20 steps, 11 cell)
                xs_out = tf.reshape(tf.concat(self.cell_outputs, 1), [-1, self.cell_size])
                
                # Define Weights and biases for each cell
                Weights_out = self.weightVar((self.cell_size, self.output_size))
                biases_out = self.biasVar((self.output_size))

                # ys_out (50 batch * steps, 1 outputs)
                self.pred = tf.nn.relu(tf.add(tf.matmul(xs_out, Weights_out), biases_out))

        # Define the loss function
        with tf.name_scope('loss'):
            reshape_pred = tf.reshape(self.pred, [self.batch_size, self.train_epoch, self.output_size])
            sum_loss = tf.reduce_mean(self.meanSquareError(reshape_pred, self.ys), 0)

            # Compute the average loss
            self.loss = tf.div(tf.reduce_sum(sum_loss, 0, name='sum_loss'), self.batch_size, name='average_loss')

        # Define the optimizer
        with tf.variable_scope('trian'):
            self.learning_rate = tf.convert_to_tensor(self.learning_rate)
            self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    @staticmethod
    def weightVar(shape, name='Weights'):
        init = tf.random_normal_initializer(mean=0., stddev=0.5)
        return tf.get_variable(shape=shape, initializer=init, name=name)


    @staticmethod
    def biasVar(shape, name='biases'):
        init = tf.constant_initializer(0.1)
        return tf.get_variable(shape=shape, initializer=init, name=name)
    

    @staticmethod
    def meanSquareError(labels, logits):
        return tf.square(tf.subtract(labels, logits))


''' WRONG to reuse parameters in train RNN '''
def wrongReuseParam(train_config, test_config):
    train_rnn1 = RNN(train_config)
    test_rnn1 = RNN(test_config)


''' No reuse parameters in train RNN '''
def noReuseParam(train_config, test_config):
    with tf.variable_scope('train_rnn'):
        train_rnn1 = RNN(train_config)
    with tf.variable_scope('test_rnn'):
        test_rnn1 = RNN(test_config)


def correctReuseParam(train_config, test_config):
    with tf.variable_scope('rnn') as scope:
        # Reuse parameters in train RNN
        train_rnn2 = RNN(train_config)
        scope.reuse_variables()
        test_rnn2 = RNN(test_config)


def main():
    # Define hyperparameters
    train_config = TrainConfig()
    test_config = TestConfig()

    # WRONG to reuse parameters in train RNN
    wrongReuseParam(train_config, test_config)

    # NO reuse parameters in train RNN 
    #noReuseParam(train_config, test_config)
    
    # CORRECT to reuse parameters in train RNN
    #correctReuseParam(train_config, test_config)

    ''' Start training '''
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


''' ENTRY POINT '''
if __name__ == "__main__":
    main()