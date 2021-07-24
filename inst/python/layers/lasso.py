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
    
class group_lasso_pen(reg.Regularizer):

    def __init__(self, la):
        self.la = la

    def __call__(self, x):
        return self.la * tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(x), 1)))

class TibLinearLasso(tf.keras.layers.Layer):
  def __init__(self, num_outputs=1, use_bias=False, la=0, name="tib_lasso"):
    super(TibLinearLasso, self).__init__()
    self.num_outputs = num_outputs
    self.la = la
    if self.num_outputs > 1:
      self.reg = group_lasso_pen(self.la)
    else:
      self.reg = tf.keras.regularizers.l2(self.la)
    self._name = name
      
  def build(self, input_shape):
    self.fc = tf.keras.layers.Dense(input_shape = input_shape, units = self.num_outputs, use_bias=False,
                                    bias_regularizer=None, activation=None, kernel_regularizer=self.reg)
    self.sc = SimplyConnected(la=self.la)


  def call(self, input):
    return self.fc(self.sc(input))

class ProjectC(tf.keras.constraints.Constraint):

    def __init__(self, C, fac):
        self.C = C
        self.fac = fac

    def __call__(self, w):
        mean = tf.reduce_mean(w)
        wold = w
        w = w - mean
        tf.debugging.assert_equal(w.shape, C.shape)
        wnew = tf.divide((w - mean), C)
        return fac*wnew + (1-fac)*wold

    def get_config(self):
        return {'C': self.C}

class TibLinearLassoConstraint(tf.keras.layers.Layer):
    def __init__(self, C, fac=0.1, num_outputs=1, use_bias=False, la=0, name="const_tib_lasso"):
        super(TibLinearLassoConstraint, self).__init__()
        self.num_outputs = num_outputs
        self.la = la
        if self.num_outputs > 1:
            self.reg = group_lasso_pen(self.la)
        else:
            self.reg = tf.keras.regularizers.l2(self.la)
        self._name = name
        self.C = C
        self.fac = fac

    def build(self, input_shape):
        self.sqrtAlpha = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer=tf.keras.initializers.RandomUniform(minval=1e-4, maxval=0.1, seed=None),
            trainable=True,
            constraint=tf.keras.constraints.NonNeg()
        )
        self.beta = self.add_weight(
            shape=(input_shape[-1], self.num_outputs),
            initializer="random_normal",
            trainable=True,
            constraint=ProjectC(self.C, self.fac)
        )

    def call(self, input):
        u = self.beta/self.sqrtAlpha
        simplyConnectedResult = tf.math.multiply(input, tf.transpose(self.sqrtAlpha))
        output = tf.matmul(simplyConnectedResult, u)
        self.add_loss(tf.reduce_sum(tf.square(u) + tf.square(self.sqrtAlpha)))
        return output
