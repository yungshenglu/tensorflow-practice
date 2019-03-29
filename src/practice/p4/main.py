#!/usr/bin/env python3

import tensorflow as tf

def main():
    ''' Create TensorFlow model '''
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)

    output = tf.multiply(input1, input2)

    ''' Start training '''
    with tf.Session() as sess:
        print(sess.run(output, feed_dict={
            input1: [28.],
            input2: [28.]
        }))


''' ENTRY POINT '''
if __name__ == "__main__":
    main()