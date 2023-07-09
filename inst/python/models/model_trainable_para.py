import tensorflow as tf
from tensorflow.keras import Model

def create_model_with_param(init_val = [1.0]):

    class ModelwP(Model):
        def __init__(self, **kwargs):
            super(ModelwP, self).__init__(**kwargs)
            self.a = tf.Variable(init_val, name="extra_param")

        def call(self, inputs, training=True, mask=None):
            return inputs
            
    return ModelwP
