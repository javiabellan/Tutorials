# Tensorflow tutorial

## Prerequisites

#### Installation

```
pip install tensorflow
```

#### Usage

```python
import tensorflow as tf
```

## Tensors

 * **Tensor** = array
 * **Tensor rank** = number of dimensions
 * **Tensor shape** = size of its dimensions


### Computational graph

TensorFlow operations are expressed into a graph of nodes.
Each node takes zero or more tensors as inputs and produces a tensor as an output.

![alt text](https://www.tensorflow.org/images/getting_started_add.png)

#### Types of nodes

 * `tf.constant(value, [dtype])`: Takes no inputs, and it outputs a value
 * `tf.Variable(initial value, [dtype])`
 * `tf.placeholder([dtype])`: Input
 * `tf.add(nodes...)`: Addition (also can be used with `+`)

#### Print node

`print(node)`

#### Evaluate node (run the model)

```python
sess = tf.Session()             # creates a Session object
print(sess.run([node1, node2])) # then invokes its run method
```


## Examples

#### Addition model

```python
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

sess = tf.Session()
print(sess.run(adder_node, {a: 3, b: 4.5}))          # 7.5
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))  # [ 3.  7.]
```

![Addition model](https://www.tensorflow.org/images/getting_started_adder.png)

#### Linear model

```python
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

# Output
# W: [-0.9999969] b: [ 0.99999082] loss: 5.69997e-11
```

