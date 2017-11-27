# Introducción

## Idea principal

Expresar una computación numérica en forma de grafo

* Los nodos representan operaciones
* Las aristas representan números (tensores)

![alt text](https://www.tensorflow.org/images/getting_started_add.png)


## ¿Qué es un tensor?

Es un conjunto de números de N dimensines. Para cada dimensión habrá que especificar el tamaño.

#### Ejemplos

* Tensor de 1 dimensión de tamaño 5: `[2, 3, 1, 2, 4]`
* Tensor de 2 dimensiones de tamaño 3x2: `[  [1, 3],  [5, 2],  [3, 4] ]`
* Tensor de 3 dimensiones de tamaño 2x2x2: `[ [[1,2], [3,4]],  [[5,6], [7,8]]  ]`

Los tensores sirven para representar datos del mundo real, por ejemplo una palabra es un tensor de 1 dimensión (las letras), una imagen es un tensor de 2 dimensiones (el alto y ancho de los píxeles), y un vídeo es un tensor de 3 dimensiones (alto, ancho y tiempo).


#### Ejemplos en tensorflow ([tf.Tensor](https://www.tensorflow.org/api_docs/python/tf/Tensor))

Los tensores en tensorflow se pueden representar de varias formas:

* Constante `tf.constant`: Si los valores no se van a modificar.
* Variable `tf.Variable`: Si los valores cambian.
* A definir `tf.placeholder`: Si el valor va a ser dado como entrada.
* Salida de una operación: Las salidas de las operacion, son tensores tabién. 



```python
tensor = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
tensor.shape()
```



## ¿Qué es un tensor?
