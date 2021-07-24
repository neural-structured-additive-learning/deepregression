#' Options for smooths setup in the pre-processing
#'
#' @param defaultSmoothing function applied to all s-terms, per default (NULL)
#' the minimum df of all possible terms is used.
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
#' @param hat1 logical; if TRUE, the smoothing parameter is defined by the trace of the hat
#' matrix sum(diag(H)), else sum(diag(2*H-HH))
#' @param sp_scale function of response; for scaling the DRO calculated penalty (1/n per default)
#' @param penalty_summary tensorflow function; summary function for the penalty in the spline layer;
#' default is \code{k_sum}. Another option could be \code{k_mean}.
#' 
#' @return Returns a list with options
#' @export
#'
smooth_control <- function(defaultSmoothing = NULL, 
                           df = 7,
                           null_space_penalty = FALSE,
                           absorb_cons = FALSE,
                           anisotropic = TRUE,
                           zero_constraint_for_smooths = TRUE,
                           hat1 = FALSE,
                           sp_scale = function(x) dim(x)[1],
                           penalty_summary = tf$keras$backend$sum,
                           variational_options = bayes_control(),
                           variational = FALSE
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
              hat1 = hat1,
              sp_scale = sp_scale,
              penalty_summary = penalty_summary,
              variational_options = variational_options,
              variational = variational))
  
}


#' Options for Fellner-Schall algorithm
#'
#' @param factor a factor defining how much of the past is used for the Fellner-Schall
#' algorithm; defaults to 0.01. 
#' @param lr_scheduler a scheduler adapting
#' \code{factor} in each step; defaults to \code{NULL}.
#' @param avg_over_past logical, whether the beta coefficients should be averaged 
#' over the past values to stabilize estimation; defaults to \code{FALSE}
#' @param constantdiv small positive constant to stabilize training
#' in small batch regimes; defaults to 0.0.
#' @param constantinv small positive constant to stabilize training
#' in small batch regimes; defaults to 0.0.
#' @param constinv_scheduler scheduler for \code{constantinv}; per default 
#' NULL which results in an exponential decay with rate 1 
#' @return Returns a list with options
#' @export
#'
fsbatch_control <- function(factor = 0.01,
                            lr_scheduler = NULL,
                            avg_over_past = FALSE,
                            constantdiv = 0,
                            constantinv = 0,
                            constinv_scheduler = NULL)
{
  
  return(list(factor = factor,
              lr_scheduler = lr_scheduler,
              avg_over_past = avg_over_past,
              constantdiv = constantdiv,
              constantinv = constantinv,
              constinv_scheduler = constinv_scheduler))
  
}


#' Options for Bayesian deepregression
#' 
#' @param posterior_fun function defining the posterior function for the variational
#' verison of the network
#' @param prior_fun function defining the prior function for the variational
#' verison of the network
#' @param kl_weight KL weights for variational networks
#' @return Returns a list with options
#' @export
#'
bayes_control <- function(posterior_fun = posterior_mean_field,
                          prior_fun = prior_trainable,
                          kl_weight = function() 1 / n_obs)
{
  
  return(list(posterior_fun = posterior_fun,
              prior_fun = prior_fun,
              kl_weight = kl_weight))
  
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
#' @param orthogonalize logical; if set to \code{FALSE}, orthogonalization is deactivated
#' @param deep_top function; optional function to put on top of the deep network instead
#' of splitting the function using \code{split_fun}
#' @return Returns a list with options
#' @export
#'
orthog_control <- function(split_fun = split_model,
                           orthog_type = c("tf", "manual"),
                           orthogonalize = TRUE,
                           deep_top = NULL)
{
  
  # check orthog type
  orthog_type <- match.arg(orthog_type)
  
  # define orthogonalization function
  orthog_fun <- switch(orthog_type,
                       col = orthog_ncol,
                       tf = orthog_tf,
                       manual = orthog)
  
  return(list(split_fun = split_fun,
              orthog_type = orthog_type,
              orthogonalize = orthogonalize,
              orthog_fun = orthog_fun,
              deep_top = deep_top))
  
}

