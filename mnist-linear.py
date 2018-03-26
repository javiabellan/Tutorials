import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# Load data
# The MNIST data-set is about 12 MB and will be downloaded automatically if it is not located in the given path.
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)
print("MNIST Data:")
print("- Training:   {}".format(len(data.train.labels)))
print("- Test:       {}".format(len(data.test.labels)))
print("- Validation: {}".format(len(data.validation.labels)))
print(data)
# mnist.train.images: [55000, 784]
# mnist.train.labels: [55000, 10]

data.test.cls = np.array([label.argmax() for label in data.test.labels])


# Data dimensions
img_size      = 28
img_size_flat = img_size * img_size
img_shape     = (img_size, img_size)
num_classes   = 10


# TensorFlow Graph
x          = tf.placeholder(tf.float32, [None, img_size_flat])
weights    = tf.Variable(tf.zeros([img_size_flat, num_classes]))
biases     = tf.Variable(tf.zeros([num_classes]))
logits     = tf.matmul(x, weights) + biases
y_pred     = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)

y_true     = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])


# Cost function
# Uses the values of the logits because it also calculates the softmax internally.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
#cross_entropy2 = - sumatorio(y_true * log(y_pred))
#cross_entropy2 = -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])
cost          = tf.reduce_mean(cross_entropy)


# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

###########################################################

# Crear la sesión
session = tf.Session()

# Iniciar variables (weights y biases)
session.run(tf.global_variables_initializer())

batch_size = 100
def optimize(num_iterations):
	for i in range(num_iterations):
		# Devulve batch_size ejemplos aleatorios
		x_batch, y_true_batch = data.train.next_batch(batch_size)

		# Poner datos en diccionario
		feed_dict_train = {x: x_batch, y_true: y_true_batch}

		# Entrenar
		session.run(optimizer, feed_dict=feed_dict_train)

###########################################################


# Medidas para ver como va
correct_prediction = tf.equal(y_pred_cls, y_true_cls) # vector de booleanos
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # la media de aciertos

feed_dict_test = {x: data.test.images,
                  y_true: data.test.labels,
                  y_true_cls: data.test.cls}


def print_accuracy():
	# Use TensorFlow to compute the accuracy.
	acc = session.run(accuracy, feed_dict=feed_dict_test)

	# Print the accuracy.
	print("Precisión para el conjunto TEST: {0:.1%}".format(acc))


def print_confusion_matrix():
    # Get the true classifications for the test-set.
    cls_true = data.test.cls
    
    # Get the predicted classifications for the test-set.
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Make various adjustments to the plot.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()


def mostrarPesos():
    # Obtener los pesos desde la varible de TensorFlow
    w = session.run(weights)
    
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i<10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            image = w[:, i].reshape(img_shape)

            # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

optimize(num_iterations=1000)
print_accuracy() # sobre 91.8%
print_confusion_matrix()
mostrarPesos()