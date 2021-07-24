#' @title Fitting Semi-Structured Deep Distributional Regression
#'
#' @param y response variable
#' @param list_of_formulas a named list of right hand side formulas,
#' one for each parameter of the distribution specified in \code{family};
#' set to \code{~ 1} if the parameter should be treated as constant.
#' Use the \code{s()}-notation from \code{mgcv} for specification of
#' non-linear structured effects and \code{d(...)} for
#' deep learning predictors (predictors in brackets are separated by commas),
#' where \code{d} can be replaced by an name name of the names in
#' \code{list_of_deep_models}, e.g., \code{~ 1 + s(x) + my_deep_mod(a,b,c)},
#' where my_deep_mod is the name of the neural net specified in
#' \code{list_of_deep_models} and \code{a,b,c} are features modeled via
#' this network.
#' @param list_of_deep_models a named list of functions
#' specifying a keras model.
#' See the examples for more details.
#' @param family a character specifying the distribution. For information on
#' possible distribution and parameters, see \code{\link{make_tfd_dist}}. Can also 
#' be a custom distribution
#' @param data data.frame or named list with input features
#' @param tf_seed a seed for tensorflow (only works with R version >= 2.2.0)
#' @param return_prepoc logical; if TRUE only the pre-processed data and layers are returned (default FALSE).
#' @param subnetwork_builder function to build each subnetwork (network for each distribution parameter;
#' per default \code{subnetwork_init})
#' @param model_builder function to build the model based on additive predictors (per default \code{keras_dr}).
#' In order to work with the methods defined for the class \code{deepregression}, the model should behave
#' like a keras model
#' @param fitting_function function to fit the instantiated model when calling \code{fit}. Per default
#' the keras \code{fit} function.
#' @param ... further arguments passed to the \code{model_builder} function
#'
#' @import tensorflow tfprobability keras mgcv dplyr R6 reticulate Matrix
#'
#' @importFrom keras fit compile
#' @importFrom tfruns is_run_active view_run_metrics update_run_metrics write_run_metadata
#' @importFrom graphics abline filled.contour matplot par points
#' @importFrom stats as.formula model.matrix terms terms.formula uniroot var dbeta coef predict
#' @importFrom methods slotNames is as
#'
#' @export
#'
#' @examples
#' library(deepregression)
#'
#' n <- 1000
#' data = data.frame(matrix(rnorm(4*n), c(n,4)))
#' colnames(data) <- c("x1","x2","x3","xa")
#' formula <- ~ 1 + deep_model(x1,x2,x3) + s(xa) + x1
#'
#' deep_model <- function(x) x %>%
#' layer_dense(units = 32, activation = "relu", use_bias = FALSE) %>%
#' layer_dropout(rate = 0.2) %>%
#' layer_dense(units = 8, activation = "relu") %>%
#' layer_dense(units = 1, activation = "linear")
#'
#' y <- rnorm(n) + data$xa^2 + data$x1
#'
#' mod <- deepregression(
#'   list_of_formulas = list(loc = formula, scale = ~ 1),
#'   data = data, y = y,
#'   list_of_deep_models = list(deep_model = deep_model)
#' )
#'
#' # train for more than 10 epochs to get a better model
#' mod %>% fit(epochs = 10, early_stopping = TRUE)
#' mod %>% fitted() %>% head()
#' cvres <- mod %>% cv()
#' mod %>% get_partial_effect(name = "s(xa)")
#' mod %>% coef()
#' mod %>% plot()
#' 
#' mod <- deepregression(
#'   list_of_formulas = list(loc = ~ 1 + s(xa) + x1, scale = ~ 1, 
#'                           dummy = ~ -1 + deep_model(x1,x2,x3) %OZ% 1),
#'   data = data, y = y,
#'   list_of_deep_models = list(deep_model = deep_model),
#'   mapping = list(1,2,1:2)
#' )
#'
deepregression <- function(
  y,
  list_of_formulas,
  list_of_deep_models,
  family = "normal",
  data,
  tf_seed = as.integer(1991-5-4),
  return_prepoc = FALSE,
  subnetwork_builder = subnetwork_init,
  model_builder = keras_dr,
  fitting_function = utils::getFromNamespace("fit.keras.engine.training.Model", "keras"),
  smooth_options = smooth_control(),
  orthog_options = orthog_control(),
  ...
)
{

  if(!is.null(tf_seed)) 
    try(tensorflow::set_random_seed(tf_seed), silent = TRUE)

  # first check if an env is available
  if(!reticulate::py_available())
  {
    message("No Python Environemt available. Use check_and_install() ",
            "to install recommended environment.")
    invisible(return(NULL))
  }

  if(!py_module_available("tensorflow"))
  {
    message("Tensorflow not available. Use install_tensorflow().")
    invisible(return(NULL))
  } # nocov end

  # convert data.frame to list
  if(is.data.frame(data)){
    data <- as.list(data)
  }

  # for convenience transform NULL to list(NULL) for list_of_deep_models
  if(missing(list_of_deep_models) | is.null(list_of_deep_models)){
    list_of_deep_models <- list(NULL)

  }else if(!is.list(list_of_deep_models)) stop("list_of_deep_models must be a list.")

  # get names of networks
  netnames <- names(list_of_deep_models)

  if(is.null(netnames) & length(list_of_deep_models) == 1)
    netnames <- "d"
  if(!is.null(list_of_deep_models) && is.null(names(list_of_deep_models)))
    stop("Please provide a named list of deep models.")
  
  if(length(netnames)>0){
    list_of_deep_models <- lapply(list_of_deep_models, dnn_placeholder_processor)
    names(list_of_deep_models) <- netnames
  }
  
  # check if user wants automatic orthogonalization
  if(orthog_options$orthogonalize){
    specials_to_oz <- netnames
    automatic_oz_check <- TRUE
  }else{
    specials_to_oz <- c()
    automatic_oz_check <- FALSE
  }
    
  # number of observations
  n_obs <- NROW(y)
  
  # number of output dim
  output_dim <- NCOL(y)

  # check for lists in list
  if(is.list(data)){
    if(any(sapply(data, class)=="list"))
      stop("Cannot deal with lists in list. Please remove list items in your data input.")
  }
  
  # check list of formulas is always one-sided
  if(any(sapply(list_of_formulas, function(x)
    attr( terms(x, data = data[!sapply(data, is.null)]) , "response" ) != 0 ))){
    stop("Only one-sided formulas are allowed in list_of_formulas.")
  }
  
  cat("Preparing additive formula(e)...")
  # parse formulas
  parsed_formulas_contents <- lapply(1:length(list_of_formulas),
                                     function(i){
                                       
                                       so <- smooth_options
                                       if(!is.null(so$df) && length(so$df)>1) so$df <- so$df[[i]]
                                       if(length(so$zero_constraint_for_smooths)>1) 
                                         so$zero_constraint_for_smooths <- 
                                           so$zero_constraint_for_smooths[i]
                                       
                                       res <- do.call("processor", 
                                                      c(list(form = list_of_formulas[[i]],
                                                           data = data,
                                                           controls = so,
                                                           output_dim = output_dim,
                                                           param_nr = i,
                                                           specials_to_oz = 
                                                             specials_to_oz, 
                                                           automatic_oz_check = 
                                                             automatic_oz_check
                                                           ),
                                                        list_of_deep_models))
                                       
                                       return(res) 
                                     })
  
  names(parsed_formulas_contents) <- names(list_of_formulas)
  
  cat(" Done.\n")

  if(return_prepoc)
    return(parsed_formulas_contents)
  
  # create additive predictor per formula
  additive_predictors <- lapply(1:length(parsed_formulas_contents), function(i)
    subnetwork_builder(parsed_formulas_contents[[i]], 
                    deep_top = orthog_options$deep_top,
                    orthog_fun = orthog_options$orthog_fun, 
                    split_fun = orthog_options$split_fun,
                    param_nr = i)
  )
    
  
  
  # initialize model
  model <- model_builder(additive_predictors, family, ...)

  ret <- list(model = model,
              init_params = 
                list(
                  list_of_formulas = list_of_formulas,
                  additive_predictors = additive_predictors,
                  parsed_formulas_contents = parsed_formulas_contents,
                  y = y,
                  ellipsis = list(...),
                  family = family,
                  smooth_options = smooth_options,
                  orthog_options = orthog_options
                ),
              fit_fun = fitting_function)


  class(ret) <- "deepregression"
  
  return(ret)

}

#' @title Define a Deep Distributional Regression Model
#'
#'
#' @param list_pred_param list of input-output(-lists) generated from
#' \code{subnetwork_init}
#' @param family see \code{?deepregression}
#' @param mapping a list of integers. The i-th list item defines which element
#' elements of \code{list_pred_param} are used for the i-th parameter.
#' For example, \code{map = list(1,2,1:2)} means that \code{list_pred_param[[1]]}
#' is used for the first distribution parameter, \code{list_pred_param[[2]]} for
#' the second distribution parameter and  \code{list_pred_param[[3]]} for both
#' distribution parameters (and then added once to \code{list_pred_param[[1]]} and
#' once to \code{list_pred_param[[2]]})
#' @param add_layer_shared_pred layer to extend shared layers defined in \code{mapping}
#' @return a list with input tensors and output tensors that can be passed
#' to, e.g., \code{keras_model}
#'
#' @export
from_preds_to_dist <- function(
  list_pred_param,
  family,
  mapping = NULL,
  add_layer_shared_pred = function(x, units) layer_dense(x, units = units, 
                                                         use_bias = FALSE)
)
{
  
  if(!is.null(mapping)){
    
    lpp <- list_pred_param
    list_pred_param <- list()
    nr_params <- max(unlist(mapping)) 
    
    if(!is.null(add_layer_shared_pred)){
      
      len_map <- sapply(mapping, length)
      multiple_param <- which(len_map>1)
      
      for(ind in multiple_param){
        # add units
        lpp[[ind]] <- tf$split(
          lpp[[ind]] %>% add_layer_shared_pred(units = len_map[ind]),
          len_map[ind],
          axis=1L
          )
        
      }
      
      lpp <- unlist(lpp, recursive = FALSE)
      mapping <- as.list(unlist(mapping))
       
    }
    
    for(i in 1:nr_params){
      list_pred_param[[i]] <- layer_add_identity(lpp[which(sapply(mapping, function(mp) i %in% mp))])
    }

    if(!is.null(names(lpp))) names(list_pred_param) <- names(lpp)[1:nr_params]
    
  }else{
  
    nr_params <- length(list_pred_param)
    
  }
  
  # check family
  if(is.character(family)){
    dist_fun <- make_tfd_dist(family)
  }else{ # assuming that family is a dist_fun already
    dist_fun <- family
  }
  nrparams_dist <- attr(dist_fun, "nrparams_dist")
  
  if(nrparams_dist < nr_params) 
  {
    warning("More formulas specified than parameters available.",
            " Will only use ", nrparams_dist, " formula(e).")
    nr_params <- nrparams_dist
    list_pred_param <- list_pred_param[1:nrparams_dist]
  }
  
  if(is.null(names(list_pred_param))){
    names(list_pred_param) <- names_families(family)
  }
  
  # concatenate predictors
  preds <- layer_concatenate_identity(unname(list_pred_param))
  
  ############################################################
  ### Define Distribution Layer ####
  
  if(family %in% c("betar", "gammar", "pareto_ls", "inverse_gamma_ls")){
    
    dist_fun <- family_trafo_funs_special(family)
    
  }
  
  # generate output
  # out <- tfprobability::layer_distribution_lambda(preds, make_distribution_fn = dist_fun)
  out <- tfp$layers$DistributionLambda(dist_fun)(preds)
  
  return(out)
  
}

#' @title Compile a Deep Distributional Regression Model
#'
#'
#' @param list_pred_param list of input-output(-lists) generated from
#' \code{subnetwork_init}
#' @param family see \code{?deepregression}
#' @param weights vector of positive values; optional (default = 1 for all observations)
#' @param ind_fun function applied to the model output before calculating the
#' log-likelihood. Per default independence is assumed by applying \code{tfd_independent}.
#' @param optimizer optimizer used. Per default Adam
#' @param model_instance which class of model to use (default \code{keras_model})
#' @param monitor_metrics Further metrics to monitor
#' @param arguments passed to other functions
#' @return a list with input tensors and output tensors that can be passed
#' to, e.g., \code{keras_model}
#'
#' @export
keras_dr <- function(
  list_pred_param,
  family,
  weights = NULL,
  ind_fun = function(x) tfd_independent(x),
  optimizer = tf$keras$optimizers$Adam(),
  model_fun = keras_model,
  monitor_metrics = list(),
  ...
)
{

  inputs <- lapply(list_pred_param, function(x) x[1:(length(x)-1)])
  outputs <- lapply(list_pred_param, function(x) x[[length(x)]])
  out <- from_preds_to_dist(outputs, family, list(...)$mapping)

  ############################################################
  ################# Define and Compile Model #################
  
  # the final model is defined by its inputs
  # and outputs

  model <- model_fun(inputs = unlist(inputs, recursive = TRUE),
                     outputs = out)

  # define weights to be equal to 1 if not given
  if(is.null(weights)) weights <- 1

  # the negative log-likelihood is given by the negative weighted
  # log probability of the model
  if(family!="pareto_ls"){  
    negloglik <- function(y, model) 
      - weights * (model %>% ind_fun() %>% tfd_log_prob(y)) 
  }else{
    negloglik <- function(y, model)
      - weights * (model %>% ind_fun() %>% tfd_log_prob(y + model$scale))
  }

  model %>% compile(optimizer = optimizer,
                    loss = negloglik,
                    metrics = monitor_metrics)
  
  return(model)

}