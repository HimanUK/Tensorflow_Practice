import tensorflow as tf

#from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
#mnist = read_data_sets("D:\Study\ML and ANN\Tensorflow practice\MNIST", one_hot = True)

#GRAPH CONSTRUCTION
hl1 = 500
hl2 = 500
hl3 = 500

n_cls = 10
batch = 100

x = tf.placeholder('float', [None,784])
y = tf.placeholder('float') #Size not specified: Any tensor can be fed 

def CNN_data(data):
    h1_layer = {'weights': tf.Variable(tf.random_normal([784,hl1])), 'biases': tf.Variable(tf.random_normal(hl1))}
    h2_layer = {'weights': tf.Variable(tf.random_normal([hl1,hl2])), 'biases': tf.Variable(tf.random_normal(hl2))}
    h3_layer = {'weights': tf.Variable(tf.random_normal([hl2,hl3])), 'biases': tf.Variable(tf.random_normal(hl3))}
    
    output_layer = {'weights': tf.Variable(tf.random_normal([hl3,n_cls])), 'biases': tf.Variable(tf.random_normal(n_cls))}
    
    l1 = tf.add(tf.matmul(data,h1_layer['weights']),h1_layer['biases'])
    l1_relu = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1_relu,h2_layer['weights']),h2_layer['biases'])
    l2_relu = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2_relu,h3_layer['weights']),h3_layer['biases'])
    l3_relu = tf.nn.relu(l3)
    
    output_final = tf.add(tf.matmul(l3_relu,output_layer['weights']),output_layer['biases'])
    output_final_relu = tf.nn.relu(output_final)
    
    return(output_final_relu)
