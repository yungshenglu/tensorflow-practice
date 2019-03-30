#!/usr/bin/env python3

import tensorflow as tf
import numpy as np


# Only show error message
tf.logging.set_verbosity(tf.logging.ERROR)


def main():
    # Notes: remember to define the same dtype and shape when restore
    weights = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=tf.float32, name='weights')
    biases = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')

    # Create a saver for saving
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize all variables in TensorFlow
        sess.run(tf.global_variables_initializer())

        # Save the varibles to the file
        save_path = saver.save(sess, './out/model.ckpt')
        print('[INFO] Save to the file: ', save_path)


''' ENTRY POINT '''
if __name__ == "__main__":
    main()