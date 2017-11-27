# Introducción

## Idea principal

Expresar una computación numérica en forma de grafo

* Los nodos representan operaciones
* Las aristas representan números (tensores)


## ¿Qué es un tensor?

Es un conjunto de números de N dimensines. Para cada dimensión habrá que especificar el tamaño.

> #### Ejemplos:
>
> * Un tensor de 1 dimensión de tamaño 5: `[2, 3, 1, 2, 4]`
> * Un tensor de 2 dimensiones de tamaño 3x2: `[  [1, 3],  [5, 2],  [3, 4] ]`
> * Un tensor de 3 dimensiones de tamaño 2x2x2: `[ [[1,2], [3,4]],  [[5,6], [7,8]]  ]`

Los tensores sirven para representar datos del mundo real, por ejemplo una palabra es un tensor de 1 dimensión (las letras), una imagen es un tensor de 2 dimensiones (el alto y ancho de los píxeles), y un vídeo es un tensor de 3 dimensiones (alto, ancho y tiempo).


## Ejemplo en tensorflow

Los tensores son los datos 

```python
tensor = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
tensor.shape()
```


