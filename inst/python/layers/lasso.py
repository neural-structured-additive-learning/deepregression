import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.regularizers as reg

# simply-connected (SC) layer used for construction
class SimplyConnected(keras.layers.Layer):
    def __init__(self, la=0, multfac_initializer=tf.initializers.Ones):
        super(SimplyConnected, self).__init__()
        self.la = la
        self.multfac_initializer = multfac_initializer
        
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], ),
            initializer=self.multfac_initializer,
            regularizer=reg.l2(self.la),
            trainable=True,
        )
        
    def call(self, inputs):
        return tf.math.multiply(inputs, self.w)
        
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'la': self.la
        })
        return config
  
    
# SC layer (diagonal with length 1) followed by FC output    
class TibLinearLasso(tf.keras.layers.Layer):
    def __init__(self, units=1, la=0, kernel_initializer=tf.keras.initializers.HeNormal, multfac_initializer=tf.initializers.Ones, **kwargs):
        super(TibLinearLasso, self).__init__(**kwargs)
        self.units = units
        self.la = la
        # if self.units > 1:
            #  self.reg = inverse_group_lasso_pen(self.la)
            # else:
        self.reg = reg.l2(self.la)
        # self._name = name
        self.kernel_initializer = kernel_initializer
        self.multfac_initializer = multfac_initializer
                
    def build(self, input_shape):
        self.fc = tf.keras.layers.Dense(input_shape = input_shape, 
                                    units = self.units, 
                                    use_bias=False,
                                    bias_regularizer=None, 
                                    activation=None, 
                                    kernel_initializer=self.kernel_initializer,
                                    kernel_regularizer=self.reg
                                    )
        self.sc = SimplyConnected(la=self.la, multfac_initializer=self.multfac_initializer)

    def call(self, input):
        return self.fc(self.sc(input))
        
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'la': self.la
        })
        return config

    
# grouping (GC) layer used for constructions  
class GroupConnected(keras.layers.Layer):
    def __init__(self, group_idx=None, la=0, multfac_initializer=tf.keras.initializers.HeNormal):
        super(GroupConnected, self).__init__()
        self.la = la
        self.input_shapes = [len(gii) for gii in group_idx]
        self.group_idx = group_idx
        self.multfac_initializer = multfac_initializer
        
    def build(self, input_shape):
        self.w = [self.add_weight(
            shape=(inps, 1),
            initializer=self.multfac_initializer,
            regularizer=reg.l2(self.la),
            trainable=True) for inps in self.input_shapes]
        
    def call(self, inputs):
        gathered_inputs = [tf.gather(inputs, ind, axis = 1) for ind in self.group_idx]
        return tf.squeeze(tf.stack([tf.matmul(gathered_inputs[i], self.w[i]) 
                          for i in range(len(gathered_inputs))], axis=1), axis=-1)
                          
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'group_idx': self.group_idx,
            'la': self.la
        })
        return config


# grouping layer followed by FC output
class TibGroupLasso(tf.keras.layers.Layer):
    def __init__(self, units=1, group_idx=None, la=0, kernel_initializer=tf.initializers.Ones, multfac_initializer=tf.keras.initializers.HeNormal, **kwargs):
        super(TibGroupLasso, self).__init__(**kwargs)
        self.units = units
        self.la = la
        self.reg = reg.l2(self.la)
        # self._name = name
        self.group_idx = group_idx
        self.kernel_initializer = kernel_initializer
        self.multfac_initializer = multfac_initializer
      
    def build(self, input_shape):
        if self.group_idx is None:
            self.fc = tf.keras.layers.Dense(units = 1, 
                                            use_bias=False, 
                                            bias_regularizer=None, 
                                            activation=None, 
                                            kernel_regularizer=self.reg,
                                            kernel_initializer=self.kernel_initializer
                                            )
            self.gc = tf.keras.layers.Dense(input_shape = input_shape, 
                                        units = 1, 
                                        use_bias=False, 
                                        bias_regularizer=None, 
                                        activation=None, 
                                        kernel_regularizer=self.reg,
                                        kernel_initializer=self.multfac_initializer
                                        )
        else:
            self.fc = tf.keras.layers.Dense(input_shape = input_shape, 
                                            units = self.units, 
                                            use_bias=False, 
                                            bias_regularizer=None, 
                                            activation=None, 
                                            kernel_regularizer=self.reg,
                                            kernel_initializer=self.kernel_initializer
                                            )
            self.gc = GroupConnected(group_idx=self.group_idx, la=self.la, multfac_initializer=self.multfac_initializer)


    def call(self, input):
        return self.fc(self.gc(input))
        
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'group_idx': self.group_idx,
            'la': self.la
        })
        return config


# diagonal layers of length (depth-1) followed by FC output
class HadamardLayer(tf.keras.layers.Layer):    
    def __init__(self, units=1, la=0, depth=2, kernel_initializer=tf.keras.initializers.HeNormal, multfac_initializer=tf.initializers.Ones, **kwargs):
        super(HadamardLayer, self).__init__(**kwargs)
        self.units = units
        self.la = la
        self.depth = depth
        self.reg = reg.l2(self.la)
        self.kernel_initializer = kernel_initializer
        self.multfac_initializer = multfac_initializer
        # self._name = name
      
    def build(self, input_shape):
        self.fc = tf.keras.layers.Dense(input_shape = input_shape, 
                                        units = self.units, 
                                        use_bias=False,
                                        bias_regularizer=None, 
                                        activation=None, 
                                        kernel_regularizer=self.reg,
                                        kernel_initializer=self.kernel_initializer
                                        )
        # create list of diagonal layers
        self.diaglayers = [SimplyConnected(la=self.la, multfac_initializer=self.multfac_initializer) for x in range(0, self.depth-1)]
    
        # use sequential model class for diagonal block
        self.diagblock = tf.keras.Sequential(self.diaglayers)
    

    def call(self, input):
        return self.fc(self.diagblock(input))  

        
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'la': self.la,
            'depth': self.depth
        })
        return config

# grouping layer followed by diagonal layers followed by FC output
class GroupHadamardLayer(tf.keras.layers.Layer):
    def __init__(self, units=1, group_idx=None, la=0, depth=3, kernel_initializer=tf.keras.initializers.HeNormal, multfac_initializer=tf.initializers.Ones, **kwargs):
        super(GroupHadamardLayer, self).__init__(**kwargs)
        self.units = units
        self.la = la
        # self._name = name
        self.reg = reg.l2(self.la)
        self.depth = depth
        self.group_idx = group_idx
        self.kernel_initializer = kernel_initializer
        self.multfac_initializer = multfac_initializer
      
    def build(self, input_shape):
        self.fc = tf.keras.layers.Dense(input_shape = input_shape, 
                                        units = self.units, 
                                        use_bias=False, 
                                        bias_regularizer=None, 
                                        activation=None, 
                                        kernel_regularizer=self.reg,
                                        kernel_initializer=self.kernel_initializer
                                        )
        self.gc = GroupConnected(group_idx=self.group_idx, la=self.la, multfac_initializer=self.multfac_initializer)
        self.diaglayers = [SimplyConnected(la=self.la, multfac_initializer=self.multfac_initializer) for x in range(0, (self.depth-2))]
        self.diagblock = tf.keras.Sequential(self.diaglayers)

    def call(self, input):
        return self.fc(self.diagblock(self.gc(input)))

        
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'group_idx': self.group_idx,
            'la': self.la,
            'depth': self.depth
        })
        return config

# explicit penalties for comparisons   
class inverse_group_lasso_pen(reg.Regularizer):
    def __init__(self, la):
        self.la = la

    def __call__(self, x):
        return self.la * tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(x), 1)))


class ExplicitGroupLasso(reg.Regularizer):    
    def __init__(self, la=0, group_idx=None, **kwargs):
        super(ExplicitGroupLasso, self).__init__(**kwargs)
        self.la = la
        self.group_idx = group_idx
        self.group_shapes = [len(gii) for gii in group_idx]
    
    def __call__(self, x):
        self.gathered_inputs = [tf.gather(x, ind, axis=0) for ind in self.group_idx]
        return self.la * tf.reduce_sum([tf.sqrt(tf.reduce_sum(tf.square(self.gathered_inputs[i]))) 
                       for i in range(len(self.gathered_inputs))])
                       

class BlownUpPenalty(reg.Regularizer):    
    def __init__(self, la=0, group_idx=None):
        super(BlownUpPenalty, self).__init__()
        self.la = la
        self.group_idx = group_idx
        self.group_shapes = [len(gii) for gii in group_idx]
    
    def __call__(self, x):
        return self.la * tf.reduce_sum(tf.multiply(tf.cast(self.group_shapes, "float32"), tf.square(x))) 
        
        
class TibGroupLassoBlownUp(tf.keras.layers.Layer):
    def __init__(self, units=1, group_idx=None, la=0, kernel_initializer=tf.keras.initializers.HeNormal, multfac_initializer=tf.initializers.Ones, **kwargs):
        super(TibGroupLassoBlownUp, self).__init__(**kwargs)
        self.units = units
        self.la = la
        # self._name = name
        self.group_idx = group_idx
        self.reg_dense = BlownUpPenalty(self.la, self.group_idx)
        self.kernel_initializer = kernel_initializer
        self.multfac_initializer = multfac_initializer
      
    def build(self, input_shape):
        self.fc = tf.keras.layers.Dense(input_shape = input_shape, 
                                        units = self.units, 
                                        use_bias=False, 
                                        bias_regularizer=None, 
                                        activation=None, 
                                        kernel_regularizer=self.reg_dense,
                                        kernel_initializer=self.kernel_initializer
                                        )
        self.gc = GroupConnected(group_idx=self.group_idx, la=self.la, multfac_initializer=self.multfac_initializer)


    def call(self, input):
        return self.fc(self.gc(input))
        
        
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'group_idx': self.group_idx,
            'la': self.la
        })
        return config
                       
                       
class HadamardDiffLayer(keras.layers.Layer):
    def __init__(self, units=1, la=0, initu='glorot_uniform', initv='glorot_uniform', **kwargs):
        super(HadamardDiffLayer, self).__init__(**kwargs)
        self.la = la
        self.initu = initu
        self.initv = initv
        self.units = units
        
    def build(self, input_shape):
        self.u = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=self.initu,
            regularizer=reg.l2(self.la),
            trainable=True,
        )
        self.v = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=self.initv,
            regularizer=reg.l2(self.la),
            trainable=True,
        )
        
    def call(self, inputs):
        beta = tf.subtract(tf.square(self.u), tf.square(self.v))
        return tf.matmul(inputs, beta)
        
        
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'la': self.la,
            'initu': self.initu,
            'initv': self.initv
        })
        return config
