import tensorflow as tf
    
#Defining input and output placeholders  
input1 = tf.placeholder(shape=[None, 3], dtype=tf.float32)  
output1 = tf.placeholder(shape=[None, 1], dtype=tf.float32) 

"""  
Weights and biases defined as TF Variables  
Variables - SIZE does not change
Placeholers - SIZE can be variable
"""
weights = tf.Variable(initial_value=[[0.3], [0.1], [0.8]], dtype=tf.float32)  
bias = tf.Variable(initial_value=[[1]], dtype=tf.float32)  
   
#Preparing inputs of the activation function  
af = tf.matmul(input1, weights) + bias
   
#Activation function definition  
sigm = tf.nn.sigmoid(af)  
   
#Measuring the prediction error  
error = tf.reduce_sum(output1 - sigm)  
   
#Minimizing the prediction error using gradient descent optimizer  
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(error)  
  
#TFsession starts 
sess = tf.Session()  
  
# Initializing the tf Variables (weights and bias)  
sess.run(tf.global_variables_initializer())  
  
# Training data input1s  
input1_data = [[255, 0, 0],  
                        [248, 80, 68],  
                        [0, 0, 255],  
                        [67, 15, 210]]  
  
# Training data desired outputs  
output1_data = [[1],  
                         [1],  
                         [0],  
                         [0]]  
 
#Training loop of the neural network  
for i in range(10000):  
    sess.run(fetches=[train_op], feed_dict={
                                  input1: input1_data,  
                                   output1: output1_data})  
  
#Class scores of some testing data  
print("Expected Scores : ", sess.run(fetches=sigm, feed_dict={input1: [[248, 80, 68],[0, 0, 255]]}))  
 
#Closing the tf Session to free resources  
sess.close()
