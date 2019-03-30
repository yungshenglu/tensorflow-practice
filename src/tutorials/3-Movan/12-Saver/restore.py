#!/usr/bin/env python3

import tensorflow as tf
import numpy as np


# Only show error message
tf.logging.set_verbosity(tf.logging.ERROR)


def main():
    # Re-define the same dtype and shape for your variables
    weights = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name='weights')
    biases = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name='biases')

    # No need the step of global_variables_initializer()!

    # Create a saver for restoring
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Restore the varibles from the file
        restore_path = './out/model.ckpt'
        saver.restore(sess, restore_path)
        print('[INFO] Restore from the file: ', restore_path)

        # Show the variables of weights and biases
        print('weights:\n', sess.run(weights))
        print('biases:\n', sess.run(biases))


''' ENTRY POINT '''
if __name__ == "__main__":
    main()