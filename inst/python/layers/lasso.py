import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.regularizers as reg

class SimplyConnected(keras.layers.Layer):
    def __init__(self, la=0):
        super(SimplyConnected, self).__init__()
        w_init = tf.random_normal_initializer()
        self.la = la
        
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], ),
            initializer="random_normal",
            regularizer=tf.keras.regularizers.l2(self.la),
            trainable=True,
        )
        
    def call(self, inputs):
        return tf.math.multiply(inputs, self.w)
    
class inverse_group_lasso_pen(reg.Regularizer):

    def __init__(self, la):
        self.la = la

    def __call__(self, x):
        return self.la * tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(x), 1)))

class TibLinearLasso(tf.keras.layers.Layer):
  def __init__(self, num_outputs=1, la=0, name="tib_lasso"):
    super(TibLinearLasso, self).__init__()
    self.num_outputs = num_outputs
    self.la = la
    if self.num_outputs > 1:
      self.reg = inverse_group_lasso_pen(self.la)
    else:
      self.reg = tf.keras.regularizers.l2(self.la)
    self._name = name
      
  def build(self, input_shape):
    self.fc = tf.keras.layers.Dense(input_shape = input_shape, units = self.num_outputs, use_bias=False,
                                    bias_regularizer=None, activation=None, kernel_regularizer=self.reg)
    self.sc = SimplyConnected(la=self.la)


  def call(self, input):
    return self.fc(self.sc(input))
