#importing libraries
import tensorflow as tf
import numpy as np

numpy_inputs = np.mat([[5,2,13],[7,9,0]], dtype = int)

inputs = tf.convert_to_tensor(value = numpy_inputs, dtype = tf.int8)

#session start
with tf.Session() as sess:
    print(sess.run(fetches = inputs))
    print(inputs)

#session end
sess.close()
