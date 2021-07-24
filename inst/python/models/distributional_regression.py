import tensorflow as tf
from psplines import psplines.fsbatch_control as fsbatch_control
from helpers import create_inputs
import tf.keras.layers.Dense as layer_dense
import tf.keras.layers.Add as layer_add
import tf.keras.layers.Concatenate as layer_concatenate
from common import list_handling.remove_nones as remove_nones

#' @title Initializing a Distributional Regression Model
#'
#'
#' @param n_obs number of observations
#' @param ncol_structured a vector of length #parameters
#' defining the number of variables used for each of the parameters.
#' If any of the parameters is not modelled using a structured part
#' the corresponding entry must be zero.
#' @param list_structured list of (non-linear) structured layers
#' (list length between 0 and number of parameters)
#' @param nr_params number of distribution parameters
#' @param family family specifying the distribution that is modelled
#' @param dist_fun a custom distribution applied to the last layer,
#' see \code{\link{make_tfd_dist}} for more details on how to construct
#' a custom distribution function.
#' @param weights observation weights used in the likelihood
#' @param monitor_metric see \code{?deepregression}
#' @param output_dim dimension of the output (> 1 for multivariate outcomes)
#' @param mixture_dist see \code{?deepregression}
#' @param ind_fun see \code{?deepregression}
#' @param extend_output_dim see \code{?deepregression}
#' @param offset list of logicals corresponding to the paramters;
#' defines per parameter if an offset should be added to the predictor
#' @param additional_penalty to specify any additional penalty, provide a function
#' that takes the \code{model$trainable_weights} as input and applies the
#' additional penalty. In order to get the correct index for the trainable
#' weights, you can run the model once and check its structure.
#' @param fsbatch_options options for Fellner-Schall algorithm, see 
#' \code{?deepregression}
#'
#' @export
def dr_init(
  n_obs,
  ncol_structured,
  list_structured,
  family,
  nr_params = 2,
  dist_fun = None,
  weights = None,
  monitor_metric = [],
  output_dim = 1,
  mixture_dist = False,
  ind_fun = lambda x: x,
  extend_output_dim = 0,
  offset = None,
  additional_penalty = None,
  fsbatch_options = fsbatch_control(),
  optimizer = tf.keras.optimizers.SGD()
):
	
	# create structured inputs
	inputs_struct = create_inputs(ncol_structured)

	# create offset			
	if offset is not None:
		offset_inputs = create_inputs(offset)
		offset_layers = create_offset(offset_inputs)

	# redefine output dimension				
	if len(extend_output_dim) > 1 or extend_output_dim is not 0:
		output_dim = output_dim + extend_output_dim
	else:
		output_dim = [output_dim]*len(inputs_struct)  
		
	# define structured predictor
	structured_parts = [] 
	
	for i in range(len(inputs_struct)):
	
		if inputs_struct[i] is None:
			structured_parts += [None]
		else:
			if list_structured[i] is None:
				structured_parts += [create_structured_linear(inp = inputs_struct[i], outdim = output_dim[i], name = "structured_linear_" + str(i))]
			else:
				this_layer = list_structured[i]
				structured_parts += [this_layer(inputs_struct[i])]
				
	# add offset to structured part
	if offset is not None:
		for i in range(len(structured_parts)):
			if offset[i] is not None and offset[i] is not 0:
				structured_parts[i] = layer_add([structured_parts[i], offset_layers[i]])
				
	# concatenate predictors -> just to split them later again?
	if len(structured_parts) > 1:
		preds = layer_concatenate(structured_parts) 
	else:
		preds = structured_parts[1]
		
	if mixture_dist:
		preds = create_mixture_preds(preds)
		
	# apply the transformation for each parameter
	# and put in the right place of the distribution
	if dist_fun is None:
		out = do_call_tfd_layer(preds, family)
	else:
		out = dist_fun(preds)
		
	############################################################
	################# Define and Compile Model #################
	
	# define all inputs
	inputList = remove_nones(inputs_struct)
	
	if offset is not None:
	
	inputList = inputList + remove_nones(offset_inputs)
	
	Plist = 
	fsbatch_options['Plist'] = Plist
	
	kerasGAM = do_call('build_kerasGAM', fsbatch_options)
	
	# the final model is defined by its inputs
	# and outputs
	model = kerasGAM(inputs = inputList, outputs = out)
	
	for i, ls in enumerate(list_structured):
		if ls[i] is not None:
			reg = ls[i].get_penalty
			model.add_loss(reg)
			
	# define weights to be equal to 1 if not given
	if weights is None:
		weights = 1
		
	# the negative log-likelihood is given by the negative weighted
	# log probability of the model
	if family != "pareto_ls":
		def negloglik(y, model):
			return(- weights * (ind_fun(model).tfd_log_prob(y)))
	else:
		def negloglik(y, model):
			return(- weights * (ind_fun(model).tfd_log_prob(y + model.scale)))
	
	if additional_penalty is not None:
		add_loss = model.add_loss(lambda x: additional_penalty(model.trainable_weights))
		
	# compile the model using the defined optimizer,
	# the negative log-likelihood as loss funciton
	# and the defined monitoring metrics as metrics
	
	model.compile(optimizer = optimizer, loss = negloglik, metrics = monitor_metric)
	
	return(model)
