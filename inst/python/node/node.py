import tensorflow as tf
#import tensorflow_addons as tfa
from activations import sparsemax 
from tensorflow_probability import distributions, stats
import numpy as np

def log_to_file(tensor, filename, tensor_name):
    with open(filename, 'a') as f:
        # Convert tensor to a numpy array and write to file
        f.write(tensor_name + ":" + str(tensor.numpy()) + '\n')

@tf.function
def sparsemoid(inputs):
    return tf.clip_by_value(0.5 * inputs + 0.5, 0., 1.)


def get_binary_lookup_table(depth):
    # output: binary tensor [depth, 2**depth, 2]
    indices = tf.keras.backend.arange(0, 2**depth, 1)
    offsets = 2 ** tf.keras.backend.arange(0, depth, 1)
    bin_codes = (tf.reshape(indices, (1, -1)) // tf.reshape(offsets, (-1, 1)) % 2)
    bin_codes = tf.stack([bin_codes, 1 - bin_codes], axis=-1)
    bin_codes = tf.cast(bin_codes, 'float32')
    binary_lut = tf.Variable(initial_value=bin_codes, trainable=False)
    return binary_lut


def get_feature_selection_logits(n_trees, depth, dim):
    initializer = tf.keras.initializers.random_uniform()
    init_shape = (dim, n_trees, depth)
    init_value = initializer(shape=init_shape, dtype='float32')
    return tf.Variable(init_value, trainable=True)


def get_output_response(n_trees, depth, units):
    initializer = tf.keras.initializers.random_uniform()
    init_shape = (n_trees, units, 2**depth)
    init_value = initializer(init_shape, dtype='float32')
    return tf.Variable(initial_value=init_value, trainable=True)


def get_feature_thresholds(n_trees, depth):
    initializer = tf.ones_initializer()
    init_shape = (n_trees, depth)
    init_value = initializer(shape=init_shape, dtype='float32')
    return tf.Variable(init_value, trainable=True)


def get_log_temperatures(n_trees, depth):
    initializer = tf.ones_initializer()
    init_shape = (n_trees, depth)
    init_value = initializer(shape=init_shape, dtype='float32')
    return tf.Variable(initial_value=init_value, trainable=True)


def init_feature_thresholds(features, beta, n_trees, depth):
    sampler = distributions.Beta(beta, beta)
    percentiles_q = sampler.sample([n_trees * depth])

    flattened_feature_values = tf.map_fn(tf.keras.backend.flatten, features)
    percentile = stats.percentile(flattened_feature_values, 100*percentiles_q)
    feature_thresholds = tf.reshape(percentile, (n_trees, depth))
    return feature_thresholds


def init_log_temperatures(features, feature_thresholds):
    input_threshold_diff = tf.math.abs(features - feature_thresholds)
    log_temperatures = stats.percentile(input_threshold_diff, 50, axis=0)
    return log_temperatures


class ObliviousDecisionTree(tf.keras.layers.Layer):
    def __init__(self,
                 n_trees=3,
                 depth=4,
                 units=1,
                 threshold_init_beta=1.):
        super(ObliviousDecisionTree, self).__init__()
        self.initialized = False
        self.n_trees = n_trees
        self.depth = depth
        self.units = units
        self.threshold_init_beta = threshold_init_beta

    # invoked by the first __call__() to the layer, and supplies the shape(s) of the input(s), which may not have been known at initialization time;
    def build(self, input_shape):
        dim = input_shape[-1]
        n_trees, depth, units = self.n_trees, self.depth, self.units
        
        # trainable parameters 
        ### initialization via random_uniform(), shape: (dim, n_trees, depth)
        self.feature_selection_logits = get_feature_selection_logits(n_trees,
                                                                     depth,
                                                                     dim)
                                                                     
        ### initialization via tf.ones_initializer(), shape: (n_trees, depth)                                                             
        self.feature_thresholds = get_feature_thresholds(n_trees, depth)

        ### initialization via tf.ones_initializer(), shape: (n_trees, depth)                                                             
        self.log_temperatures = get_log_temperatures(n_trees, depth)

        ### initialization via random_uniform(), (n_trees, units, 2^depth)
        self.response = get_output_response(n_trees, depth, units)

        # non-trainable parameter, binary tensor [depth, 2**depth, 2]
        self.binary_lut = get_binary_lookup_table(depth)

    
    # assigns starting values to self.feature_thresholds and self.log_temperatures
    def _data_aware_initialization(self, inputs):
        beta, n_trees, depth = self.threshold_init_beta, self.n_trees, self.depth

        feature_values = self._get_feature_values(inputs)
        feature_thresholds = init_feature_thresholds(feature_values, beta, n_trees, depth)
        log_temperatures = init_log_temperatures(feature_values, feature_thresholds)
        
        self.feature_thresholds.assign(feature_thresholds)
        self.log_temperatures.assign(log_temperatures)

    def _get_feature_values(self, inputs, training=None):
        # Probability per feature, per tree, per level of tree (i,n,d)
        feature_selectors = sparsemax(self.feature_selection_logits)
        
        # shape: (batch_size, n_trees, tree_depth)
        # b = batch, i = feature, n = n_tress, d = depth
        # per batch data point, tree und tree level weighted sum of input features with the corresponding weights from feature_selectors
        feature_values = tf.einsum('bi,ind->bnd', inputs, feature_selectors)
        return feature_values

    def _get_feature_gates(self, feature_values):
        threshold_logits = (feature_values - self.feature_thresholds)
        threshold_logits = threshold_logits * tf.math.exp(-self.log_temperatures)
        
        threshold_logits = tf.stack([-threshold_logits, threshold_logits], axis=-1)

        feature_gates = sparsemoid(threshold_logits)
        return feature_gates

    def _get_aggregated_response(self, feature_gates):
        # b: batch, n: number of trees, d: depth of trees, s: 2 (binary channels)
        # c: 2**depth, u: units (response units)
        
        aggregated_gates = tf.einsum('bnds,dcs->bndc', feature_gates, self.binary_lut)
        aggregated_gates = tf.math.reduce_prod(aggregated_gates, axis=-2)
       
        # response per batch-DP, tree and unit
        # weighted linear combination of response tensor entries with weights from the entries of choice tensor C
        # shape (b, n, u)
        aggregated_response = tf.einsum('bnc,nuc->bnu', aggregated_gates, self.response)
        return aggregated_response

 
    def call(self, inputs, training=None):
        if not self.initialized:
            self._data_aware_initialization(inputs)
            self.initialized = True
            
        feature_values = self._get_feature_values(inputs)
        feature_gates = self._get_feature_gates(feature_values)
        aggregated_response = self._get_aggregated_response(feature_gates)
        
        # shape: (b, u)
        response_averaged_over_trees = tf.reduce_mean(aggregated_response, axis=1)
        return response_averaged_over_trees


class NODE(tf.keras.Model):
    def __init__(self,
                 units = 1,
                 n_layers = 1,
                 n_trees = 1,
                 tree_depth = 1,
                 threshold_init_beta = 1):

        super(NODE, self).__init__()
        self.units = units
        self.n_layers = n_layers
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.threshold_init_beta = threshold_init_beta
        
        self.bn = tf.keras.layers.BatchNormalization()
        self.ensemble = [ObliviousDecisionTree(n_trees=n_trees,
                             depth=tree_depth,
                             units=units,
                             threshold_init_beta=threshold_init_beta)
                         for _ in range(n_layers)]


# defines forward pass 
# general procedure: 
# 1) batch normalization of complete input
# 2) call ODT Layers using the input and the output from the previous layer
    def call(self, inputs, training=None):
        x = self.bn(inputs, training=training)
        for tree in self.ensemble:
            h = tree(x)
            x = tf.concat([x, h], axis=1)
        return h
      

def layer_node(units, n_layers, n_trees, tree_depth, threshold_init_beta):
    return(NODE(units, n_layers, n_trees, tree_depth, threshold_init_beta))
