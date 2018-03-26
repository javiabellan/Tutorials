import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math




# Load data
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)


# Training Parameters
learning_rate = 0.001
num_steps = 200
batch_size = 128
display_step = 10

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_clases = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


# Datos a alimentar
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_clases])


# Variables de la red
pesos = {
    'c1': tf.Variable(tf.random_normal([5, 5, 1, 32])),     # 5x5 conv: 1 input, 32 outputs
    'c2': tf.Variable(tf.random_normal([5, 5, 32, 64])),    # 5x5 conv: 32 inputs, 64 outputs
    'd1': tf.Variable(tf.random_normal([7*7*64, 1024])),    # densa: 7*7*64 inputs, 1024 outputs
    'd2': tf.Variable(tf.random_normal([1024, num_clases])) # densa: 1024 inputs, 10 outputs (class prediction)
}

bias = {
    'c1': tf.Variable(tf.random_normal([32])),
    'c2': tf.Variable(tf.random_normal([64])),
    'd1': tf.Variable(tf.random_normal([1024])),
    'd2': tf.Variable(tf.random_normal([num_clases]))
}

######################################################################## Grafo de Tensorflow

def conv2d(x, W, b, strides=1, name="conv"):
	with tf.name_scope(name):
		# Conv2D wrapper, with bias and relu activation
		x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
		x = tf.nn.bias_add(x, b)
		return tf.nn.relu(x)

def maxpool2d(x, k=2, name="pool"):
	with tf.name_scope(name):
		# MaxPool2D wrapper
		return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def aplana(x):
	# Get the shape of the input layer.
	#layer_shape = x.get_shape()
	#num_features = layer_shape[1:4].num_elements()
	#layer_flat = tf.reshape(x, [-1, num_features])
	#return layer_flat
	return tf.reshape(x, [-1, pesos['d1'].get_shape().as_list()[0]])
	#return tf.reshape(x, [-1, 7*7*64])

def densa(x, W, b):
	x = tf.matmul(x, W)
	x = tf.add(x, b)
	x = tf.nn.relu(x)
	x = tf.nn.dropout(x, dropout)
	return x



#g = tf.Graph()
#with g.as_default():
entrada = tf.reshape(X, shape=[-1, 28, 28, 1], name='entrada')
conv1   = conv2d(entrada, pesos['c1'], bias['c1'], name='conv1')
pool1   = maxpool2d(conv1, k=2, name='pool1')
conv2   = conv2d(pool1, pesos['c2'], bias['c2'], name='conv2')
pool2   = maxpool2d(conv2, k=2, name='pool2')
aplanar = aplana(pool2)
densa1  = densa(aplanar, pesos['d1'], bias['d1'])
salida  = tf.add(tf.matmul(densa1, pesos['d2']), bias['d2'])
predicción = tf.nn.softmax(salida)

'''
Capa datos:     (?, 784)
Capa entrada:   (?, 28, 28, 1)
Capa convol 1:  (?, 28, 28, 32)
Capa pooling 1: (?, 14, 14, 32)
Capa convol 2:  (?, 14, 14, 64)
Capa pooling 1: (?, 7, 7, 64)
Capa aplana:    (?, 12544) # Debería ser 7*7*64 = 3136
'''

print("TENSORES DE LA RED NEURONAL")
print("Capa datos:    ", X.get_shape())
print("Capa entrada:  ", entrada.get_shape())
print("Capa convol 1: ", conv1.get_shape())
print("Capa pooling 1:", pool1.get_shape())
print("Capa convol 2: ", conv2.get_shape())
print("Capa pooling 1:", pool2.get_shape())
print("Capa aplana:   ", aplanar.get_shape())
print("Capa densa1:   ", densa1.get_shape())
print("Capa salida:   ", salida.get_shape())
print("Capa predicción: ", predicción.get_shape())


# create a graph...
#tf.summary.FileWriter("logs/cnn", g).close()

sess = tf.Session()
writer = tf.summary.FileWriter("logs/cnn", sess.graph).close()

########################################################### Entrenar

'''
# Define cost and optimizer
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=salida, labels=Y)
cost          = tf.reduce_mean(cross_entropy)
optimizer     = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op      = optimizer.minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(predicción, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                      Y: mnist.test.labels[:256],
                                      keep_prob: 1.0}))
'''