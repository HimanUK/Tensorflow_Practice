import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

hl1 = 500
hl2 = 500
hl3 = 500

n_cls = 10
batch = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def TFvertices(data):
    h1l = {'weights':tf.Variable(tf.random_normal([784, hl1])),'biases':tf.Variable(tf.random_normal([hl1]))}

    h2l = {'weights':tf.Variable(tf.random_normal([hl1,hl2])), 'biases':tf.Variable(tf.random_normal([hl2]))}

    h3l = {'weights':tf.Variable(tf.random_normal([hl2,hl3])), 'biases':tf.Variable(tf.random_normal([hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([hl3, n_cls])),
                    'biases':tf.Variable(tf.random_normal([n_cls])),}


    l1 = tf.add(tf.matmul(data,h1l['weights']), h1l['biases'])
    l1_relu = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1_relu,h2l['weights']), hl2['biases'])
    l2_relu = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2_relu,h3l['weights']), h3l['biases'])
    l3_relu = tf.nn.relu(l3)

    output = tf.matmul(l3_relu,output_layer['weights']) + output_layer['biases']

    return output

def TFsession(x):
    predict = TFvertices(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = predict,labels = y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    epochs = 1000
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for i in range(epochs):
            loss = 0
            for _ in range(int(mnist.train.num_examples/batch)):
                ex, ey = mnist.train.next_batch(batch)
                _, c = sess.run([optimizer, cost], feed_dict={x: ex, y: ey})
                loss += c

            print('Epoch', i, 'completed out of',epochs,'loss:',loss)

        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

TFsession(x)
