import tensorflow as tf
import numpy as np
import math
from tensorflow import keras
import tensorflow.keras.regularizers as regularizers

def tf_repeat(a, dim):
    return tf.reshape(tf.tile(tf.expand_dims(a, axis = -1),  (1, 1, dim)), 
                      shape = (-1, a.shape[1]*dim))


def tf_row_tensor_left_part(a,b):
    return tf_repeat(a, b.shape[1])

def tf_row_tensor_right_part(a,b):
    return tf.tile(b, (1, a.shape[1]))

def tf_row_tensor_fun(a,b):
    return tf.multiply(tf_row_tensor_left_part(a,b), tf_row_tensor_right_part(a,b))
        
class RowTensor(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RowTensor, self).__init__(**kwargs)
 
    def call(self, inputs):
        return tf_row_tensor_fun(inputs[0],inputs[1])
        
