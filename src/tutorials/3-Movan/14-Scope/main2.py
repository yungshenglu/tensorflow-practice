#!/usr/bin/env python3

import tensorflow as tf


def main():
    ''' Create TensorFlow model '''
    # Define a variable scope named "a_variable_scope"
    with tf.variable_scope('a_variable_scope') as scope:
       # Use "tf.get_variable" to define variable
       init = tf.constant_initializer(value=3)
       var3 = tf.get_variable(name='var3', shape=[1], dtype=tf.float32, initializer=init)

       # Use "tf.Variable" to define variable
       var4 = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)
       var41 = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)

        # Reuse the variable "var3"
       scope.reuse_variables()
       var31 = tf.get_variable(name='var3')

    ''' Start training '''
    with tf.Session() as sess:
        # Initialize all variables in TensorFlow
        sess.run(tf.global_variables_initializer())

        # Show all variables above
        print(var3.name)
        print(sess.run(var3))
        print(var4.name)
        print(sess.run(var4))
        print(var41.name)
        print(sess.run(var41))
        print(var31.name)
        print(sess.run(var31))


''' ENTRY POINT '''
if __name__ == "__main__":
    main()