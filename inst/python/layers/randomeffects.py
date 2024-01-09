import tensorflow as tf
import numpy as np

class RELayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(RELayer, self).__init__(**kwargs)
        self.units = units
        self.b = self.add_weight(name='random_intercept',
                                 shape=(units,),
                                 initializer=tf.keras.initializers.RandomNormal,
                                 trainable=True)
        self.logtau = self.add_weight(name='ri_variance',
                                      shape=(),
                                      initializer='zeros',
                                      trainable=True)

    def call(self, inputs, training=False):
        # Compute Zb
        output = tf.matmul(inputs, tf.reshape(self.b, (-1, 1)))

        if True:
            # Compute the log-density
            tau_squared = tf.square(tf.math.exp(self.logtau))
            log_density = 0.5 * tf.reduce_sum(tf.square(self.b) / tau_squared) \
                          + 0.5 * self.units * tf.math.log(2 * np.pi * tau_squared)

            # Add the log-density to the loss
            self.add_loss(log_density)

        return output
