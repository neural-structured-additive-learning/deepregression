import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.regularizers as reg

# simply-connected (SC) layer used for construction
class SimplyConnected(keras.layers.Layer):
    def __init__(self, la=0):
        super(SimplyConnected, self).__init__()
        self.la = la
        
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], ),
            initializer="random_normal",
            regularizer=reg.l2(self.la),
            trainable=True,
        )
        
    def call(self, inputs):
        return tf.math.multiply(inputs, self.w)
  
    
# SC layer (diagonal with length 1) followed by FC output    
class TibLinearLasso(tf.keras.layers.Layer):
    def __init__(self, units=1, la=0, name="tib_lasso"):
        super(TibLinearLasso, self).__init__()
        self.units = units
        self.la = la
        # if self.units > 1:
            #  self.reg = inverse_group_lasso_pen(self.la)
            # else:
        self.reg = reg.l2(self.la)
        self._name = name
                
    def build(self, input_shape):
        self.fc = tf.keras.layers.Dense(input_shape = input_shape, 
                                    units = self.units, 
                                    use_bias=False,
                                    bias_regularizer=None, 
                                    activation=None, 
                                    kernel_regularizer=self.reg
                                    )
        self.sc = SimplyConnected(la=self.la)

    def call(self, input):
        return self.fc(self.sc(input))

    
# grouping (GC) layer used for constructions  
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
            regularizer=reg.l2(self.la),
            trainable=True) for inps in self.input_shapes]
        
    def call(self, inputs):
        gathered_inputs = [tf.gather(inputs, ind, axis = 1) for ind in self.group_idx]
        return tf.squeeze(tf.stack([tf.matmul(gathered_inputs[i], self.w[i]) 
                          for i in range(len(gathered_inputs))], axis=1), axis=-1)


# grouping layer followed by FC output
class TibGroupLasso(tf.keras.layers.Layer):
    def __init__(self, units=1, group_idx=None, la=0, name="tib_grouplasso"):
        super(TibGroupLasso, self).__init__()
        self.units = units
        self.la = la
        self.reg = reg.l2(self.la)
        self._name = name
        self.group_idx = group_idx
      
    def build(self, input_shape):
        self.fc = tf.keras.layers.Dense(input_shape = input_shape, 
                                        units = self.units, 
                                        use_bias=False, 
                                        bias_regularizer=None, 
                                        activation=None, 
                                        kernel_regularizer=self.reg
                                        )
        self.gc = GroupConnected(group_idx=self.group_idx, la=self.la)


    def call(self, input):
        return self.fc(self.gc(input))


# diagonal layers of length (depth-1) followed by FC output
class HadamardLayer(tf.keras.layers.Layer):    
  def __init__(self, units=1, la=0, depth=2, name="hadamard_layer"):
    super(HadamardLayer, self).__init__()
    self.units = units
    self.la = la
    self.depth = depth
    self.reg = reg.l2(self.la)
    self._name = name
      
  def build(self, input_shape):
    self.fc = tf.keras.layers.Dense(input_shape = input_shape, 
                                    units = self.units, 
                                    use_bias=False,
                                    bias_regularizer=None, 
                                    activation=None, 
                                    kernel_regularizer=self.reg
                                    )
    # create list of diagonal layers
    self.diaglayers = [SimplyConnected(la=self.la) for x in range(0, self.depth-1)]
    
    # use sequential model class for diagonal block
    self.diagblock = tf.keras.Sequential(self.diaglayers)
    

  def call(self, input):
    return self.fc(self.diagblock(input))  


# grouping layer followed by diagonal layers followed by FC output
class GroupHadamardLayer(tf.keras.layers.Layer):
    def __init__(self, units=1, group_idx=None, la=0, depth=3, name="group_hadamard"):
        super(GroupHadamardLayer, self).__init__()
        self.units = units
        self.la = la
        self._name = name
        self.reg = reg.l2(self.la)
        self.depth = depth
        self.group_idx = group_idx
      
    def build(self, input_shape):
        self.fc = tf.keras.layers.Dense(input_shape = input_shape, 
                                        units = self.units, 
                                        use_bias=False, 
                                        bias_regularizer=None, 
                                        activation=None, 
                                        kernel_regularizer=self.reg
                                        )
        self.gc = GroupConnected(group_idx=self.group_idx, la=self.la)
        self.diaglayers = [SimplyConnected(la=self.la) for x in range(0, (self.depth-2))]
        self.diagblock = tf.keras.Sequential(self.diaglayers)

    def call(self, input):
        return self.fc(self.diagblock(self.gc(input)))


# explicit penalties for comparisons   
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
        self.group_shapes = [len(gii) for gii in group_idx]
    
    def __call__(self, x):
        self.gathered_inputs = [tf.gather(x, ind, axis=0) for ind in self.group_idx]
        return self.la * tf.reduce_sum([tf.sqrt(tf.reduce_sum(tf.square(self.gathered_inputs[i]))) 
                       for i in range(len(self.gathered_inputs))])
                       
                       
class HadamardDiffLayer(keras.layers.Layer):
    def __init__(self, units=1, la=0, initu='glorot_uniform', initv='glorot_uniform'):
        super(HadamardDiffLayer, self).__init__()
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
