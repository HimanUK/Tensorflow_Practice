# import tensorflow
import tensorflow as tf
import numpy as np

# build computational graph
a = tf.placeholder(tf.int16, shape = (2,2))
b = tf.placeholder(tf.int16, shape = (2,2))

addition = tf.add(a, b)

# initialize variables
init = tf.global_variables_initializer()
rand_array = np.random.rand(2,2)

# create session and run the graph
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(addition, feed_dict={a: rand_array, b: rand_array}))

# close session
sess.close()
