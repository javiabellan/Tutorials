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

 * `tf.constant`: Takes no inputs, and it outputs a value
 * `tf.add`: Addition

#### Print node

`print(node)`

#### Evaluate node

```python
sess = tf.Session()
print(sess.run([node1, node2]))
```

---

