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

---

