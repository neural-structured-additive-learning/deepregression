# helper funs tf
tfe <- function(x) tf$math$exp(x)
tfsig <- function(x) tf$math$sigmoid(x)
tfsoft <- function(x) tf$math$softmax(x)
tfsqrt <- function(x) tf$math$sqrt(x)
tfsq <- function(x) tf$math$square(x)
tfdiv <- function(x,y) tf$math$divide(x,y)
tfrec <- function(x) tf$math$reciprocal(x)
tfmult <- function(x,y) tf$math$multiply(x,y)

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
#'  \item{"bernoulli_prob": }{bernoulli distribution with probabilities (sigmoid)}
#'  \item{"beta": }{beta with concentration 1 = alpha (exp) and concentration
#'  0 = beta (exp)}
#'  \item{"betar": }{beta with mean (sigmoid) and scale (sigmoid)}
#'  \item{"cauchy": }{location (identity), scale (exp)}
#'  \item{"chi2": }{cauchy with df (exp)}
#'  \item{"chi": }{cauchy with df (exp)}
#'  \item{"exponential": }{exponential with lambda (exp)}
#'  \item{"gamma": }{gamma with concentration (exp) and rate (exp)}
#'  \item{"gammar": }{gamma with location (exp) and scale (exp)}
#'  \item{"gumbel": }{gumbel with location (identity), scale (exp)}
#'  \item{"half_cauchy": }{half cauchy with location (identity), scale (exp)}
#'  \item{"half_normal": }{half normal with scale (exp)}
#'  \item{"horseshoe": }{horseshoe with scale (exp)}
#'  \item{"inverse_gamma": }{inverse gamma with concentation (exp) and rate (exp)}
#'  \item{"inverse_gamma_ls": }{inverse gamma with location (exp) and variance (1/exp)}
#'  \item{"inverse_gaussian": }{inverse Gaussian with location (exp) and concentation
#'  (exp)}
#'  \item{"laplace": }{Laplace with location (identity) and scale (exp)}
#'  \item{"log_normal": }{Log-normal with location (identity) and scale (exp) of
#'  underlying normal distribution}
#'  \item{"logistic": }{logistic with location (identity) and scale (exp)}
#'  \item{"negbinom": }{neg. binomial with count (exp) and prob (sigmoid)}
#'  \item{"negbinom_ls": }{neg. binomail with mean (exp) and clutter factor (exp)}
#'  \item{"pareto": }{Pareto with concentration (exp) and scale (1/exp)} 
#'  \item{"pareto_ls": }{Pareto location scale version with mean (exp) 
#'  and scale (exp), which corresponds to a Pareto distribution with parameters scale = mean
#'  and concentration = 1/sigma, where sigma is the scale in the pareto_ls version.}
#'  \item{"poisson": }{poisson with rate (exp)}
#'  \item{"poisson_lograte": }{poisson with lograte (identity))}
#'  \item{"student_t": }{Student's t with df (exp)}
#'  \item{"student_t_ls": }{Student's t with df (exp), location (identity) and
#'  scale (exp)}
#'  \item{"uniform": }{uniform with upper and lower (both identity)}
#'  \item{"zinb": }{Zero-inflated negative binomial with mean (exp), 
#'  variance (exp) and prob (sigmoid)}
#'  \item{"zip":  }{Zero-inflated poisson distribution with mean (exp) and prob (sigmoid)}
#' }
#' @param add_const small positive constant to stabilize calculations
#' @param trafo_list list of transformations for each distribution parameter.
#' Per default the transformation listed in details is applied.
#' @param output_dim number of output dimensions of the response (larger 1 for
#' multivariate case)
#'
#' @export
#' @rdname dr_families
make_tfd_dist <- function(family, add_const = 1e-8, output_dim = 1L,
                          trafo_list = NULL)
{

  tfd_dist <- family_to_tfd(family)

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
                 "zipf"
  ) | grepl("multivariate", family) | grepl("vector", family))
  stop("Family ", family, " not implemented yet.")

  if(family=="binomial")
    stop("Family binomial not implemented yet.",
         " If you are trying to model independent",
         " draws from a bernoulli distribution, use family='bernoulli'.")
  
  if(is.null(trafo_list)) trafo_list <- family_to_trafo(family)

  # check if still NULL, then probably wrong family
  if(is.null(trafo_list))
    stop("Family not implemented.")
  
  if(family=="multinomial"){

    ret_fun <- function(x) tfd_dist(trafo_list[[1]](x))

  }else if(family=="multinoulli"){

    ret_fun <- function(x) tfd_dist(trafo_list[[1]](x))

  }else{
    
    ret_fun <- create_family(tfd_dist, trafo_list, output_dim)
    
  }

  attr(ret_fun, "nrparams_dist") <- length(trafo_list)

  return(ret_fun)

}

#' Function to create (custom) family
#' 
#' @param tfd_dist a tensorflow probability distribution
#' @param trafo_list list of transformations h for each parameter 
#' (e.g, \code{exp} for a variance parameter)
#' @param output_dim integer defining the size of the response
#' @return a function that can be used by 
#' \code{tfp$layers$DistributionLambda} to create a new 
#' distribuional layer
#' @export
#' 
create_family <- function(tfd_dist, trafo_list, output_dim = 1L)
{
  
  if(length(output_dim)==1){

    # the usual  case    
    ret_fun <- function(x) do.call(tfd_dist,
                                   lapply(1:(x$shape[[2]]/output_dim),
                                          function(i)
                                            trafo_list[[i]](
                                              tf_stride_cols(x,(i-1L)*output_dim+1L,
                                                             (i-1L)*output_dim+output_dim)))
    ) 
  
  }else{
    
    # tensor-shaped output (assuming the last dimension to be 
    # the distribution parameter dimension if tfd_dist has multiple arguments)
    dist_dim <- length(trafo_list)
    ret_fun <- function(x) do.call(tfd_dist,
                                   lapply(1:(x$shape[[length(x$shape)]]/dist_dim),
                                          function(i)
                                            trafo_list[[i]](
                                              tf_stride_last_dim_tensor(x,(i-1L)*dist_dim+1L,
                                                                        (i-1L)*dist_dim+dist_dim)))
    ) 
    
  }
  
  attr(ret_fun, "nrparams_dist") <- length(trafo_list)
  
  return(ret_fun)
  
}

#' Returns the parameter names for a given family
#' 
#' @param family character specifying the family as defined by \code{deepregression}
#' @return vector of parameter names
#' 
#' @export
#' 
names_families <- function(family)
{
  
  nams <- switch(family,
                 normal = c("location", "scale"),
                 bernoulli = "logits",
                 bernoulli_prob = "probabilities",
                 beta = c("concentration", "concentration"),
                 betar = c("location", "scale"),
                 cauchy = c("location", "scale"),
                 chi2 = "df",
                 chi = "df",
                 exponential = "rate",
                 gamma = c("concentration", "rate"),
                 gammar = c("location", "scale"),
                 gumbel = c("location", "scale"),
                 half_cauchy = c("location", "scale"),
                 half_normal = "scale",
                 horseshoe = "scale",
                 inverse_gamma = c("concentation", "rate"),
                 inverse_gamma_ls = c("location", "scale"),
                 inverse_gaussian = c("location", "concentation"),
                 laplace = c("location", "scale"),
                 log_normal = c("location", "scale"),
                 logistic = c("location", "scale"),
                 multinomial = c("probs"),
                 multinoulli = c("logits"),
                 negbinom = c("count", "prob"),
                 negbinom_ls = c("mean", "clutter_factor"),
                 pareto = c("concentration", "scale"),
                 pareto_ls = c("location", "scale"),
                 poisson = "rate",
                 poisson_lograte = "lograte",
                 student_t = "df",
                 student_t_ls = c("df", "location", "scale"),
                 uniform = c("upper", "lower"),
                 zinb = c("mean", "variance", "prob"),
                 zip = c("mean", "prob")
  )
  
  return(nams)
  
}

#' Character-tfd mapping function
#' 
#' @param family character defining the distribution
#' @return a tfp distribution
#' @export
family_to_tfd <- function(family)
{
  
  # define dist_fun
  tfd_dist <- switch(family,
                     normal = tfd_normal,
                     bernoulli = tfd_bernoulli,
                     bernoulli_prob = function(probs)
                       tfd_bernoulli(probs = probs),
                     beta = tfd_beta,
                     betar = tfd_beta,
                     binomial = tfd_binomial,
                     categorical = tfd_categorical,
                     cauchy = tfd_cauchy,
                     chi2 = tfd_chi2,
                     chi = tfd_chi,
                     dirichlet_multinomial = tfd_dirichlet_multinomial,
                     dirichlet = tfd_dirichlet,
                     exponential = tfd_exponential,
                     gamma_gamma = tfd_gamma_gamma,
                     gamma = tfd_gamma,
                     gammar = tfd_gamma,
                     geometric = tfd_geometric,
                     gumbel = tfd_gumbel,
                     half_cauchy = tfd_half_cauchy,
                     half_normal = tfd_half_normal,
                     horseshoe = tfd_horseshoe,
                     inverse_gamma = tfd_inverse_gamma,
                     inverse_gamma_ls = tfd_inverse_gamma,
                     inverse_gaussian = tfd_inverse_gaussian,
                     kumaraswamy = tfd_kumaraswamy,
                     laplace = tfd_laplace,
                     log_normal = tfd_log_normal,
                     logistic = tfd_logistic,
                     mse = tfd_mse,
                     multinomial = function(probs)
                       tfd_multinomial(total_count = 1L,
                                       probs = probs),
                     multinoulli = function(logits)#function(probs)
                       # tfd_multinomial(total_count = 1L,
                       #                 logits = logits),
                       tfd_one_hot_categorical(logits),
                     # tfd_categorical,#(probs = probs),
                     negbinom = function(fail, probs)
                       tfd_negative_binomial(total_count = fail, probs = probs#,
                                             # validate_args = TRUE
                       ),
                     negbinom_ls = tfd_negative_binomial_ls,
                     pareto = tfd_pareto,
                     pareto_ls = tfd_pareto,
                     poisson = tfd_poisson,
                     poisson_lograte = function(log_rate)
                       tfd_poisson(log_rate = log_rate),
                     student_t = function(x)
                       tfd_student_t(df=x,loc=0,scale=1),
                     student_t_ls = tfd_student_t,
                     truncated_normal = tfd_truncated_normal,
                     uniform = tfd_uniform,
                     von_mises_fisher = tfd_von_mises_fisher,
                     von_mises = tfd_von_mises,
                     zinb = tfd_zinb,
                     zip = tfd_zip
                     # zipf = function(x)
                     #   tfd_zipf(x,
                     #            dtype = tf$float32,
                     #            sample_maximum_iterations =
                     #              tf$constant(100, dtype="float32"))
  )
  
  return(tfd_dist)
  
}

#' Character-to-transformation mapping function
#' 
#' @param family character defining the distribution
#' @param add_const see \code{\link{make_tfd_dist}}
#' @return a list of transformation for each distribution parameter
#' @export
family_to_trafo <- function(family, add_const = 1e-8)
{
  
  trafo_list <- switch(family,
                       normal = list(function(x) x,
                                     function(x) tf$add(add_const, tfe(x))),
                       bernoulli = list(function(x) x),
                       bernoulli_prob = list(function(x) tfsig(x)),
                       beta = list(function(x) tf$add(add_const, tfe(x)),
                                   function(x) tf$add(add_const, tfe(x))),
                       betar = list(function(x) x,
                                    function(x) x),
                       binomial = list(), # tbd
                       categorial = list(), #tbd
                       cauchy = list(function(x) x,
                                     function(x) tf$add(add_const, tfe(x))),
                       chi2 = list(function(x) tf$add(add_const, tfe(x))),
                       chi = list(function(x) tf$add(add_const, tfe(x))),
                       dirichlet_multinomial = list(), #tbd
                       dirichlet = list(), #tbd
                       exponential = list(function(x) tf$add(add_const, tfe(x))),
                       gamma_gamma = list(), #tbd
                       gamma = list(function(x) tf$add(add_const, tfe(x)),
                                    function(x) tf$add(add_const, tfe(x))),
                       geometric = list(function(x) x),
                       gammar = list(function(x) x,
                                     function(x) x),
                       gumbel = list(function(x) x,
                                     function(x) tf$add(add_const, tfe(x))),
                       half_cauchy = list(function(x) x,
                                          function(x) tf$add(add_const, tfe(x))),
                       half_normal = list(function(x) tf$add(add_const, tfe(x))),
                       horseshoe = list(function(x) tf$add(add_const, tfe(x))),
                       inverse_gamma = list(function(x) tf$add(add_const, tfe(x)),
                                            function(x) tf$add(add_const, tfe(x))),
                       inverse_gamma_ls = list(function(x) tf$add(add_const, tfe(x)),
                                               function(x) tf$add(add_const, tfe(x))),
                       inverse_gaussian = list(function(x) tf$add(add_const, tfe(x)),
                                               function(x)
                                                 tf$add(add_const, tfe(x))),
                       kumaraswamy = list(), #tbd
                       laplace = list(function(x) x,
                                      function(x) tf$add(add_const, tfe(x))),
                       log_normal = list(function(x) x,
                                         function(x) tf$add(add_const, tfe(x))),
                       logistic = list(function(x) x,
                                       function(x) tf$add(add_const, tfe(x))),
                       negbinom = list(function(x) tf$add(add_const, tfe(x)),
                                       function(x) tf$math$sigmoid(x)),
                       negbinom_ls = list(function(x) tf$add(add_const, tfe(x)),
                                          function(x) tf$add(add_const, tfe(x))),
                       multinomial = list(function(x) tfsoft(x)),
                       multinoulli = list(function(x) x),
                       mse = list(function(x) x),
                       pareto = list(function(x) tf$add(add_const, tfe(x)),
                                     function(x) add_const + tfe(-x)),
                       pareto_ls = list(function(x) tf$add(add_const, tfe(x)),
                                        function(x) tf$add(add_const, tfe(x))),
                       poisson = list(function(x) tf$add(add_const, tfe(x))),
                       poisson_lograte = list(function(x) x),
                       student_t = list(function(x) x),
                       student_t_ls = list(function(x) tf$add(add_const, tfe(x)),
                                           function(x) x,
                                           function(x) tf$add(add_const, tfe(x))),
                       truncated_normal = list(), # tbd
                       uniform = list(function(x) x,
                                      function(x) x),
                       von_mises = list(function(x) x,
                                        function(x) tf$add(add_const, tfe(x))),
                       zinb = list(function(x) tf$add(add_const, tfe(x)),
                                   function(x) tf$add(add_const, tfe(x)),
                                   function(x) tf$stack(list(tf$math$sigmoid(x),
                                                             tf$subtract(1,tf$math$sigmoid(x))),
                                                        axis=2L)),
                       zip = list(function(x) tf$add(add_const, tfe(x)),
                                  function(x) tf$stack(list(tf$math$sigmoid(x),
                                                            tf$subtract(1,tf$math$sigmoid(x))),
                                                       axis=2L)),
                       zipf = list(function(x) 1 + tfe(x))
  )
  
  return(trafo_list)
  
}

family_trafo_funs_special <- function(family, add_const = 1e-8)
{

  # specially treated distributions
  trafo_fun <- switch(family,
    gammar = function(x){

      # rate = 1/((sigma^2)*mu)
      # con = (1/sigma^2)

      mu = tfe(tf_stride_cols(x,1))
      sig = tfe(tf_stride_cols(x,2))
      sigsq = tfsq(sig)
      con = tfrec(sigsq + add_const)
      rate = tfrec(tfmult(mu, sigsq))

      return(list(concentration = con, rate = rate))
    },
    betar = function(x){

      # mu=a/(a+b)
      # sig=(1/(a+b+1))^0.5
      mu = tfsig(tf_stride_cols(x,1))
      sigsq = tfsq(tfsig(tf_stride_cols(x,2)))
      #a = tf$compat$v2$maximum(tfmult(mu, (tfrec(sigsq) - 1)), 1 + add_const)
      #b = tf$compat$v2$maximum(tfmult((tfrec(mu) - 1), a), 1 + add_const)
      a = tf$compat$v2$maximum(
        tfmult(mu, tfdiv(tf$subtract(tf$constant(1), sigsq), sigsq)),
        tf$constant(0) + add_const)
      b = tf$compat$v2$maximum(
        tfmult(a, tfdiv(tf$subtract(tf$constant(1), mu),mu)),
        tf$constant(0) + add_const)

      return(list(concentration1 = a, concentration0 = b))
    },
    pareto_ls = function(x){
      
      # k_print_tensor(x, message = "This is x")
      scale = add_const + tfe(tf_stride_cols(x,1))
      # k_print_tensor(scale, message = "This is scale")
      con = tfe(-tf_stride_cols(x,2))
      # k_print_tensor(con, message = "This is con")
      return(list(concentration = con, scale = scale)) 
      
      
    },
    inverse_gamma_ls = function(x){
      
      # alpha = 1/sigma^2
      alpha = add_const + tfe(-tf_stride_cols(x,2))
      # beta = mu (alpha + 1)
      beta = add_const + tfe(tf_stride_cols(x,1)) * (alpha + 1)
      
      return(list(concentration = alpha, scale = beta)) 
      
      
    }
  )

  tfd_dist <- switch(family,
                     betar = tfd_beta,
                     gammar = tfd_gamma,
                     pareto_ls = tfd_pareto,
                     inverse_gamma_ls = tfd_inverse_gamma
  )

  ret_fun <- function(x) do.call(tfd_dist, trafo_fun(x))
  
  attr(ret_fun, "nrparams_dist") <- 2L

  return(ret_fun)

}

#' Implementation of a zero-inflated poisson distribution for TFP
#'
#' @param lambda scalar value for rate of poisson distribution
#' @param probs vector of probabilites of length 2 (probability for poisson and
#' probability for 0s)
tfd_zip <- function(lambda, probs)
{

  return(
    tfd_mixture(cat = tfd_categorical(probs = probs),
                components =
                  list(tfd_poisson(rate = lambda),
                       tfd_deterministic(loc = lambda * 0L)
                  ),
                name="zip")
  )
}

tfd_negative_binomial_ls = function(mu, r){

  # sig2 <- mu + (mu*mu / r)
  # count <- r
  probs <- #1-tf$compat$v2$clip_by_value(
    tf$divide(r, tf$add(r, mu))#,
    # 0, 1
  # )
  
  return(tfd_negative_binomial(total_count = r, probs = probs))

}

#' Implementation of a zero-inflated negbinom distribution for TFP
#'
#' @param mu,r parameter of the negbin_ls distribution
#' @param probs vector of probabilites of length 2 (probability for poisson and
#' probability for 0s)
tfd_zinb <- function(mu, r, probs)
{

  return(
    tfd_mixture(cat = tfd_categorical(probs = probs),
                components =
                  list(tfd_negative_binomial_ls(mu = mu, r = r),
                       tfd_deterministic(loc = mu * 0L)
                  ),
                name="zinb")
  )
}

#' For using mean squared error via TFP
#' 
#' @param mean parameter for the mean
#' @details \code{deepregression} allows to train based on the
#' MSE by using \code{loss = "mse"} as argument to \code{deepregression}.
#' This tfd function just provides a dummy \code{family}
#' 
tfd_mse <- function(mean)
{
  return(
    tfd_normal(loc = mean, scale = 1)
  )
}