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
#' @param optimizer see \code{?deepregression} 
#'
#' @export
dr_init <- function(
  n_obs,
  ncol_structured,
  list_structured,
  nr_params = 2,
  family,
  dist_fun = NULL,
  weights = NULL,
  monitor_metric = list(),
  output_dim = 1,
  mixture_dist = FALSE,
  ind_fun = function(x) x,
  extend_output_dim = 0,
  offset = NULL,
  additional_penalty = NULL,
  fsbatch_options = fsbatch_control(),
  optimizer = tf$keras$optimizers$SGD()
)
{
  
  out <- from_preds_to_dist(list_pred_param, family)
  
  ############################################################
  ################# Define and Compile Model #################
  
  kerasGAM <- do.call("build_kerasGAM", fsbatch_options)
  
  # the final model is defined by its inputs
  # and outputs
  model <- kerasGAM(inputs = unlist(list_pred_param, recursive = FALSE),
                    outputs = out)
  
  l <- 1  
  for(i in 1:length(list_structured)){
    
    if(!is.null(list_structured[[i]])){
      reg <- list_structured[[i]]$get_penalty
      
      # lambdas <- list_structured[[i]]$mask * tf$exp(
      #     list_structured[[i]]$lambdas)/
      #   tf$cast(list_structured[[i]]$n, "float32")
      # P <- list_structured[[i]]$P
      # bigP = as.matrix(
      #   bdiag(lapply(1:length(lambdas),
      #                function(j) as.matrix(lambdas[j]*tf$cast(
      #                  P[j-1]$to_dense(), dtype = "float32")))))
      # layer_nr <- grep("pen_linear", 
      #                   sapply(1:length(model$trainable_weights), function(k) 
      #                     model$trainable_weights[[k]]$name))[l]
      # reg <- function(x){ 
      #   return(
      #     tf$reduce_sum(tf_incross(
      #       model$trainable_weights[[layer_nr]],
      #       tf$cast(bigP, "float32")))
      #   )
      # }
      model$add_loss(reg)
      l <- l + 1
    }
  }
  
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
  
  
  if(!is.null(additional_penalty)){
    
    add_loss <- function(x) additional_penalty(
      model$trainable_weights
    )
    model$add_loss(add_loss)
    
  }
  
  # compile the model using the defined optimizer,
  # the negative log-likelihood as loss funciton
  # and the defined monitoring metrics as metrics
  model %>% compile(optimizer = optimizer,
                    loss = negloglik,
                    metrics = monitor_metric)
  
  return(model)
  
}
