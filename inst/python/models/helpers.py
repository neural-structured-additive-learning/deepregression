import tensorflow as tf
import tf.keras.layers.InputLayer as layer_input
import tf.keras.layers.Dense as layer_dense
import tf.keras.layers.Add as layer_add
import tf.keras.layers.Concatenate as layer_concatenate

def create_inputs(list_shapes):

	inputs = []
	for i in range(len(list_shapes)):
		nc = list_shapes[i]
		if nc==0 or nc is None: 
			inputs += [None] 
		else:
			inputs += [layer_input(shape = (int(nc))]
			
	return(inputs)
	
def create_offset(offset_inputs):

	ones_initializer = tf.keras.initializers.Ones()
	
	offset_layers = []
	for i in range(len(offset_inputs)):
	
		x = offset_inputs[i]
		if x is None:
			offset_layers += [None] 
		else:
			offset_layers += [layer_dense(units = 1, activation = "linear", use_bias = False, trainable = False, kernel_initializer = ones_initializer)(x)]
			
	return(offset_layers)
	
def create_structured_linear(inp, outdim, name):

	return(layer_dense(units = int(outdim), activation = "linear", use_bias = False, name = name)(inp))
	
def create_mixture_preds(preds, dim):

	mix_prob = preds[:,0]
	rest = preds[:,1:]
	return(layer_concatenate([layer_dense(units = int(dim), activation = "softmax", use_bias = False)(mix_prob), rest]))
	
def 
