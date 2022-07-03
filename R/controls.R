#' Options for penalty setup in the pre-processing
#'
#' @param defaultSmoothing function applied to all s-terms, per default (NULL)
#' the minimum df of all possible terms is used. Must be a function the smooth term
#' from mgcv's smoothCon and an argument \code{df}.
#' @param df degrees of freedom for all non-linear structural terms (default = 7);
#' either one common value or a list of the same length as number of parameters;
#' if different df values need to be assigned to different smooth terms,
#' use df as an argument for \code{s()}, \code{te()} or \code{ti()}
#' @param null_space_penalty logical value;
#' if TRUE, the null space will also be penalized for smooth effects.
#' Per default, this is equal to the value give in \code{variational}.
#' @param absorb_cons logical; adds identifiability constraint to the basisi.
#' See \code{?mgcv::smoothCon} for more details.
#' @param anisotropic whether or not use anisotropic smoothing (default is TRUE)
#' @param zero_constraint_for_smooths logical; the same as absorb_cons,
#' but done explicitly. If true a constraint is put on each smooth to have zero mean. Can
#' be a vector of \code{length(list_of_formulas)} for each distribution parameter.
#' @param no_linear_trend_for_smooths logical; see \code{zero_constraint_for_smooths}, but
#' this removes the linear trend from splines
#' @param hat1 logical; if TRUE, the smoothing parameter is defined by the trace of the hat
#' matrix sum(diag(H)), else sum(diag(2*H-HH))
#' @param sp_scale function of response; for scaling the penalty (1/n per default)
#' 
#' @return Returns a list with options
#' @export
#'
penalty_control <- function(defaultSmoothing = NULL, 
                           df = 10,
                           null_space_penalty = FALSE,
                           absorb_cons = FALSE,
                           anisotropic = TRUE,
                           zero_constraint_for_smooths = TRUE,
                           no_linear_trend_for_smooths = FALSE,
                           hat1 = FALSE,
                           sp_scale = function(x)
                             ifelse(is.list(x) | is.data.frame(x), 1/NROW(x[[1]]), 1/NROW(x))
)
{
  
  if(is.null(defaultSmoothing))
    defaultSmoothing <- function(st, df) defaultSmoothingFun(st, df, 
                                                             hat1 = hat1,
                                                             sp_scale = sp_scale,
                                                             null_space_penalty = 
                                                               null_space_penalty,
                                                             anisotropic = 
                                                               anisotropic)
  
  return(list(defaultSmoothing = defaultSmoothing,
              df = df,
              null_space_penalty = null_space_penalty,
              absorb_cons = absorb_cons,
              anisotropic = anisotropic,
              zero_constraint_for_smooths = zero_constraint_for_smooths,
              no_linear_trend_for_smooths = no_linear_trend_for_smooths,
              hat1 = hat1,
              sp_scale = sp_scale))
  
}

#' Options for orthogonalization
#' 
#' @param split_fun a function separating the deep neural network in two parts
#' so that the orthogonalization can be applied to the first part before
#' applying the second network part; per default, the function \code{split_model} is
#' used which assumes a dense layer as penultimate layer and separates the network
#' into a first part without this last layer and a second part only consisting of a
#' single dense layer that is fed into the output layer
#' @param orthog_type one of two options; If \code{"manual"}, 
#' the QR decomposition is calculated before model fitting, 
#' otherwise (\code{"tf"}) a QR is calculated in each batch iteration via TF.
#' The first only works well for larger batch sizes or ideally batch_size == NROW(y).
#' @param orthogonalize logical; if set to \code{TRUE}, automatic orthogonalization is activated
#' @param identify_intercept whether to orthogonalize the deep network w.r.t. the intercept
#' to make the intercept identifiable
#' @param deep_top function; optional function to put on top of the deep network instead
#' of splitting the function using \code{split_fun}
#' @return Returns a list with options
#' @export
#'
orthog_control <- function(split_fun = split_model,
                           orthog_type = c("tf", "manual"),
                           orthogonalize = options()$orthogonalize,
                           identify_intercept = options()$identify_intercept,
                           deep_top = NULL)
{
  
  # check orthog type
  orthog_type <- match.arg(orthog_type)
  
  # define orthogonalization function
  orthog_fun <- switch(orthog_type,
                       tf = orthog_tf,
                       manual = orthog)
  
  return(list(split_fun = split_fun,
              orthog_type = orthog_type,
              orthogonalize = orthogonalize,
              identify_intercept = identify_intercept,
              orthog_fun = orthog_fun,
              deep_top = deep_top))
  
}

#' Options for weights of layers
#' 
#' @param len integer; the length of \code{list_of_formulas}
#' @param specific_weight_options specific options for certain
#' weight terms; must be a list of length \code{length(list_of_formulas)} and
#' each element in turn a named list (names are term names as in the formula)
#' with specific options in a list
#' @param general_weight_options default options for layers
#' @param warmstart_weights While all keras layer options are availabe,
#' the user can further specify a list for each distribution parameter
#' with list elements corresponding to term names with values as vectors
#' corresponding to start weights of the respective weights
#' @param shared_layers list for each distribution parameter;
#' each list item can be again a list of character vectors specifying terms
#' which share layers
#' 
#' @return Returns a list with options
#' 
#' 
#' 
#' @export
#'
weight_control <- function(
  specific_weight_options = NULL,
  general_weight_options = list(
    activation = NULL,
    use_bias = FALSE,
    trainable = TRUE,
    kernel_initializer = "glorot_uniform",
    bias_initializer = "zeros",
    kernel_regularizer = NULL,
    bias_regularizer = NULL,
    activity_regularizer = NULL,
    kernel_constraint = NULL,
    bias_constraint = NULL
  ),
  warmstart_weights = NULL,
  shared_layers = NULL
){
  
  
  
  if(!is.null(specific_weight_options) && !is.null(warmstart_weights)){
    
    if(!is.null(shared_layers) && 
      (length(specific_weight_options)!=length(warmstart_weights) | 
         length(specific_weight_options)!=length(shared_layers)))
      stop("specific options must be a list of the same length.")
    
    len <- length(specific_weight_options)
    
  }else if(!is.null(specific_weight_options)){
    
    len <- length(specific_weight_options)
    warmstart_weights <- vector("list", length = len)
    
  }else if(!is.null(warmstart_weights)){
    
    len <- length(warmstart_weights)
    specific_weight_options <- vector("list", length = len)
    
  }else{ # both NULL
    
    if(!is.null(shared_layers)){
      
      len <- length(shared_layers)
      warmstart_weights <- vector("list", length = len)
      specific_weight_options <- vector("list", length = len)
      
    }else{
      
      len <- 1 # unclear at this stage how many formula terms
      
    }
    
  }

  ret_list <- list(list(specific = NULL, general = NULL, 
                        warmstarts = NULL, shared_layers = NULL))[rep(1, len)]
  
  for(i in 1:length(ret_list)){
  
    ret_list[[i]]$specific <- specific_weight_options[[i]]
    ret_list[[i]]$general <- general_weight_options
    ret_list[[i]]$warmstarts <- warmstart_weights[[i]]
    ret_list[[i]]$shared_layers <- shared_layers[[i]]
    
  }
  
  return(ret_list)
  
}

#' Options for formula parsing
#' 
#' @param precalculate_gamparts logical; if TRUE (default), additive parts are pre-calculated
#' and can later be used more efficiently. Set to FALSE only if no smooth effects are in the 
#' formula(s) and a formula is very large so that extracting all terms takes long or might fail
#' @param check_form logical; if TRUE (default), the formula is checked in \code{process_terms}
#' @return Returns a list with options
#' @export
#'
form_control <- function(
    precalculate_gamparts = TRUE,
    check_form = TRUE
)
{
  
  return(
    list(precalculate_gamparts = precalculate_gamparts,
         check_form = check_form)
  )
  
}
