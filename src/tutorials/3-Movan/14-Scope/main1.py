#!/usr/bin/env python3

import tensorflow as tf


def main():
    ''' Create TensorFlow model '''
    # Define name scope named "a_name_scope"
    with tf.name_scope('a_name_scope'):
       # Use "tf.get_variable" to define variable
       init = tf.constant_initializer(value=1)
       var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=init)

       # Use "tf.Variable" to define variable
       var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
       var21 = tf.Variable(name='var2', initial_value=[2.1], dtype=tf.float32)
       var22 = tf.Variable(name='var2', initial_value=[2.2], dtype=tf.float32)
    
    ''' Start training '''
    with tf.Session() as sess:
        # Initialize all variables in TensorFlow
        sess.run(tf.global_variables_initializer())

        # Show all variables above
        print(var1.name)            # var1:0
        print(sess.run(var1))       # [1.]
        print(var2.name)            # a_name_scope/var2:0
        print(sess.run(var2))       # [2.]
        print(var21.name)           # a_name_scope/var2_1:0
        print(sess.run(var21))      # [2.1]
        print(var22.name)           # a_name_scope/var2_2:0
        print(sess.run(var22))      # [2.2]


''' ENTRY POINT '''
if __name__ == "__main__":
    main()