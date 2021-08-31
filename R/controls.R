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
#' @param hat1 logical; if TRUE, the smoothing parameter is defined by the trace of the hat
#' matrix sum(diag(H)), else sum(diag(2*H-HH))
#' @param sp_scale function of response; for scaling the penalty (1/n per default)
#' 
#' @return Returns a list with options
#' @export
#'
penalty_control <- function(defaultSmoothing = NULL, 
                           df = 7,
                           null_space_penalty = FALSE,
                           absorb_cons = FALSE,
                           anisotropic = TRUE,
                           zero_constraint_for_smooths = TRUE,
                           hat1 = FALSE,
                           sp_scale = function(x) 1/NROW(x)
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
                       tf = orthog_tf,
                       manual = orthog)
  
  return(list(split_fun = split_fun,
              orthog_type = orthog_type,
              orthogonalize = orthogonalize,
              orthog_fun = orthog_fun,
              deep_top = deep_top))
  
}
