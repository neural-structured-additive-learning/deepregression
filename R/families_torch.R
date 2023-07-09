
#' Families for deepregression
#'
#' @param family character vector
#'
#' @details
#' To specify a custom distribution, define the a function as follows
#' \code{
#' function(x) do.call(your_tfd_dist, lapply(1:ncol(x)[[1]],
#'                                     function(i)
#'                                      your_trafo_list_on_inputs[[i]](
#'                                        x[,i,drop=FALSE])))
#' }
#' and pass it to \code{deepregression} via the \code{dist_fun} argument.
#' Currently the following distributions are supported
#' with parameters (and corresponding inverse link function in brackets):
#'
#' \itemize{
#'  \item{"normal": }{normal distribution with location (identity), scale (exp)}
#'  \item{"bernoulli": }{bernoulli distribution with logits (identity)}
#'  \item{"exponential": }{exponential with lambda (exp)}
#'  \item{"gamma": }{gamma with concentration (exp) and rate (exp)}
#'  \item{"poisson": }{poisson with rate (exp)}
#' @param add_const small positive constant to stabilize calculations
#' @param trafo_list list of transformations for each distribution parameter.
#' Per default the transformation listed in details is applied.
#' @param output_dim number of output dimensions of the response (larger 1 for
#' multivariate case) (not implemented yet)
#'
#' @export
#' @rdname dr_families
make_torch_dist <- function(family, add_const = 1e-8, output_dim = 1L,
                            trafo_list = NULL){
  
  torch_dist <- family_to_trochd(family)
  
  # families not yet implemented
  if(family%in%c("categorical",
                 "dirichlet_multinomial",
                 "dirichlet",
                 "gamma_gamma",
                 "geometric",
                 "kumaraswamy",
                 "truncated_normal",
                 "von_mises",
                 "von_mises_fisher",
                 "wishart",
                 "zipf", "beta",
                 "betar",
                 "cauchy",
                 "chi2",
                 "chi",
                 "exponential",
                 "gammar",
                 "gumbel",
                 "half_cauchy",
                 "half_normal",
                 "horseshoe",
                 "inverse_gamma",
                 "inverse_gaussian",
                 "laplace",
                 "log_normal",
                 "logistic",
                 "multinomial",
                 "multinoulli",
                 "negbinom",
                 "pareto_ls",
                "poisson_lograte",
                "student_t",
                "student_t_ls",
                 "uniform",
                 "zip") | grepl("multivariate", family) | grepl("vector", family))
    stop("Family ", family, " not implemented yet.")
  
  if(family=="binomial")
    stop("Family binomial not implemented yet.",
         " If you are trying to model independent",
         " draws from a bernoulli distribution, use family='bernoulli'.")
  
  if(is.null(trafo_list)) trafo_list <- family_to_trafo_torch(family)
  
  # check if still NULL, then probably wrong family
  if(is.null(trafo_list))
    stop("Family not implemented.")
  
  ret_fun <- create_family_torch(torch_dist, trafo_list, output_dim)
  
  attr(ret_fun, "nrparams_dist") <- length(trafo_list)
  
  return(ret_fun)
  
}

#' Character-torch mapping function
#' 
#' @param family character defining the distribution
#' @return a torch distribution
#' @export
family_to_trochd <- function(family){
  # define dist_fun
  torchd_dist <- switch(family,
                        normal = distr_normal,
                        bernoulli = function(logits)
                          distr_bernoulli(logits = logits),
                        bernoulli_prob = distr_bernoulli,
                        gamma = distr_gamma,
                        poisson = distr_poisson)
}

#' Character-to-transformation mapping function
#' 
#' @param family character defining the distribution
#' @param add_const see \code{\link{make_torch_dist}}
#' @return a list of transformation for each distribution parameter
#' @export
#' 
family_to_trafo_torch <- function(family, add_const = 1e-8){
  
  trafo_list <- switch(family,
                       normal = list(function(x) x,
                                     function(x) torch_add(add_const, torch_exp(x))),
                       bernoulli = list(function(x) x),
                       bernoulli_prob = list(function(x) torch_sigmoid(x)),
                       gamma = list(
                         function(x) torch_add(add_const, torch_exp(x)),
                         function(x) torch_add(add_const, torch_exp(x))),
                       poisson = list(function(x) torch_add(add_const, torch_exp(x)))
  )
  
  return(trafo_list)
  
}

#' Function to create (custom) family
#' 
#' @param torch_dist a torch probability distribution
#' @param trafo_list list of transformations h for each parameter 
#' (e.g, \code{exp} for a variance parameter)
#' @param output_dim integer defining the size of the response
#' @return a function that can be used to train a 
#' distribution learning model in torch
#' @export
#' 
create_family_torch <- function(torch_dist, trafo_list, output_dim = 1L)
{
  
  if(length(output_dim)==1){
    
    # the usual  case    
    ret_fun <- function(x) do.call(torch_dist,
                                   lapply(1:length(x),
                                          function(i)
                                            trafo_list[[i]](x[[i]]))
    ) 
    
  }
  #else{
  
  # tensor-shaped output (assuming the last dimension to be 
  # the distribution parameter dimension if tfd_dist has multiple arguments)
  # dist_dim <- length(trafo_list)
  #  ret_fun <- function(x) do.call(tfd_dist,
  #                                lapply(1:(x$shape[[length(x$shape)]]/dist_dim),
  #                                     function(i)
  #                                       trafo_list[[i]](
  #                                        tf_stride_last_dim_tensor(x,(i-1L)*dist_dim+1L,
  #                                                                   (i-1L)*dist_dim+dist_dim)))
  #) 
  
  #}
  
  attr(ret_fun, "nrparams_dist") <- length(trafo_list)
  
  return(ret_fun)
  
}


