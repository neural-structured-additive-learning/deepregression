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

class ExplicitGroupLasso(reg.Regularizer):
    
    def __init__(self, la=0, group_idx=None):
        super(ExplicitGroupLasso, self).__init__()
        self.la = la
        self.group_idx = group_idx
        self.input_shapes = [len(gii) for gii in group_idx]
    
    def __call__(self, x):
        self.gathered_inputs = [tf.gather(x, ind, axis=0) for ind in self.group_idx]
        return self.la * tf.reduce_sum([tf.sqrt(tf.reduce_sum(tf.square(self.gathered_inputs[i]))) 
                       for i in range(len(self.gathered_inputs))])
    
class TibLinearLasso(tf.keras.layers.Layer):
  def __init__(self, units=1, la=0, name="tib_lasso"):
    super(TibLinearLasso, self).__init__()
    self.units = units
    self.la = la
    # if self.units > 1:
    #  self.reg = inverse_group_lasso_pen(self.la)
    # else:
    self.reg = tf.keras.regularizers.l2(self.la)
    self._name = name
      
  def build(self, input_shape):
    self.fc = tf.keras.layers.Dense(input_shape = input_shape, units = self.units, use_bias=False,
                                    bias_regularizer=None, activation=None, kernel_regularizer=self.reg)
    self.sc = SimplyConnected(la=self.la)


  def call(self, input):
    return self.fc(self.sc(input))
    
    
class GroupConnected(keras.layers.Layer):
    def __init__(self, group_idx=None, la=0):
        super(GroupConnected, self).__init__()
        self.la = la
        self.input_shapes = [len(gii) for gii in group_idx]
        self.group_idx = group_idx
        
    def build(self, input_shape):
        self.w = [self.add_weight(
            shape=(inps, 1),
            initializer="random_normal",
            regularizer=tf.keras.regularizers.l2(self.la),
            trainable=True) for inps in self.input_shapes]
        
    def call(self, inputs):
        gathered_inputs = [tf.gather(inputs, ind, axis = 1) for ind in self.group_idx]
        return tf.squeeze(tf.stack([tf.matmul(gathered_inputs[i], self.w[i]) 
                          for i in range(len(gathered_inputs))], axis=1), axis=-1)


class TibGroupLasso(tf.keras.layers.Layer):
    def __init__(self, units=1, group_idx=None, la=0, name="tib_grouplasso"):
        super(TibGroupLasso, self).__init__()
        self.units = units
        self.la = la
        self._name = name
        self.group_idx = group_idx
      
    def build(self, input_shape):
        self.fc = tf.keras.layers.Dense(input_shape = input_shape, units = self.units, 
                                        use_bias=False, bias_regularizer=None, activation=None, 
                                        kernel_regularizer=tf.keras.regularizers.l2(self.la))
        self.sc = GroupConnected(group_idx=self.group_idx, la=self.la)


    def call(self, input):
        return self.fc(self.sc(input))
