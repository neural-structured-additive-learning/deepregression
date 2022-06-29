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
#' @param list_of_deep_models a named list of functions specifying a keras model.
#' See the examples for more details.
#' @param family a character specifying the distribution. For information on
#' possible distribution and parameters, see \code{\link{make_tfd_dist}}. Can also
#' be a custom distribution.
#' @param data data.frame or named list with input features
#' @param tf_seed a seed for TensorFlow (only works with R version >= 2.2.0)
#' @param additional_processors a named list with additional processors to convert the formula(s).
#' Can have an attribute \code{"controls"} to pass additional controls
#' @param return_prepoc logical; if TRUE only the pre-processed data and layers are returned 
#' (default FALSE).
#' @param subnetwork_builder function to build each subnetwork (network for each distribution parameter;
#' per default \code{subnetwork_init}). Can also be a list of the same size as
#' \code{list_of_formulas}.
#' @param model_builder function to build the model based on additive predictors 
#' (per default \code{keras_dr}). In order to work with the methods defined for the class 
#' \code{deepregression}, the model should behave like a keras model
#' @param fitting_function function to fit the instantiated model when calling \code{fit}. Per default
#' the keras \code{fit} function.
#' @param penalty_options options for smoothing and penalty terms defined by \code{\link{penalty_control}}
#' @param orthog_options options for the orthgonalization defined by \code{\link{orthog_control}}
#' @param verbose logical; whether to print progress of model initialization to console
#' @param weight_options options for layer weights defined by \code{\link{weight_control}}
#' @param formula_options options for formula parsing (mainly used to make calculation more efficiently)
#' @param output_dim dimension of the output, per default 1L
#' @param ... further arguments passed to the \code{model_builder} function
#'
#' @import tensorflow tfprobability keras mgcv dplyr R6 reticulate Matrix
#'
#' @importFrom keras fit compile
#' @importFrom tfruns is_run_active view_run_metrics update_run_metrics write_run_metadata
#' @importFrom graphics abline filled.contour matplot par points
#' @importFrom stats as.formula model.matrix terms terms.formula uniroot var dbeta coef predict na.omit
#' @importFrom methods slotNames is as
#'
#' @references
#' Ruegamer, D. et al. (2021):
#' deepregression: a Flexible Neural Network Framework for Semi-Structured Deep Distributional Regression.
#' \url{https://arxiv.org/abs/2104.02705}.
#'
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
#' if(!is.null(mod)){
#'
#' # train for more than 10 epochs to get a better model
#' mod %>% fit(epochs = 10, early_stopping = TRUE)
#' mod %>% fitted() %>% head()
#' cvres <- mod %>% cv()
#' mod %>% get_partial_effect(name = "s(xa)")
#' mod %>% coef()
#' mod %>% plot()
#'
#' }
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
  list_of_deep_models = NULL,
  family = "normal",
  data,
  tf_seed = as.integer(1991-5-4),
  return_prepoc = FALSE,
  subnetwork_builder = subnetwork_init,
  model_builder = keras_dr,
  fitting_function = utils::getFromNamespace("fit.keras.engine.training.Model", "keras"),
  additional_processors = list(),
  penalty_options = penalty_control(),
  orthog_options = orthog_control(),
  weight_options = weight_control(),
  formula_options = form_control(),
  output_dim = 1L,
  verbose = FALSE,

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
  }

  # convert data.frame to list
  if(is.data.frame(data)){
    data <- as.list(data)
  }

  # for convenience transform NULL to list(NULL) for list_of_deep_models
  if(missing(list_of_deep_models) | is.null(list_of_deep_models)){
    list_of_deep_models <- list(d = NULL)
    netnames <- NULL
  }else{
    # get names of networks
    netnames <- names(list_of_deep_models)

    if(is.null(netnames) & length(list_of_deep_models) == 1)
    {
      names(list_of_deep_models) <- netnames <- "d"
    }
    if(!is.null(list_of_deep_models) && is.null(names(list_of_deep_models)))
      stop("Please provide a named list of deep models.")
  }

  # create list for image variables
  # (and overwrite it potentially later)
  image_var <- list()

  if(length(netnames)>0){

    len_dnns <- sapply(list_of_deep_models, length)

    # check for image dnns
    if(any(len_dnns>1)){

      image_var <- lapply(list_of_deep_models[len_dnns>1], "[[", 2)

    }

    names(image_var) <- netnames[len_dnns>1]
    list_of_deep_models <- lapply(list_of_deep_models, dnn_processor)
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
  if(is.character(family) && (family=="multinoulli" | family=="multinomial"))
    output_dim <- NCOL(y)

  # check for lists in list
  if(is.list(data)){
    if(any(sapply(data, class)=="list"))
      stop("Cannot deal with lists in list. Please remove list items in your data input.")
  }

  # check list of formulas is always one-sided
  if(any(sapply(list_of_formulas, length)>2)){
    stop("Only one-sided formulas are allowed in list_of_formulas.")
  }

  # check for further controls
  if(!is.null(attr(additional_processors, "controls")))
    penalty_options <- c(penalty_options, attr(additional_processors, "controls"))

  # repeat weight options if not specified otherwise
  if(length(weight_options)!=length(list_of_formulas))
    weight_options <- weight_options[rep(1, length(list_of_formulas))]

  # training mse
  is.lfun <- is.function(list(...)$loss)
  if (!is.lfun) {
    if (!is.null(list(...)$loss) && list(...)$loss=="mse") {
      weight_options[[2]]$specific <- c(weight_options[[2]]$specific,
                                        list("1" = list(trainable = FALSE)))
      weight_options[[2]]$warmstarts <- c(weight_options[[2]]$warmstarts,
                                          list("1" = 0))
      family <- "normal"
    }
  }

  if(verbose) cat("Pre-calculate GAM terms...")
  so <- penalty_options
  if(formula_options$precalculate_gamparts)
    so$gamdata <- precalc_gam(list_of_formulas, data, so) else
      so$gamdata <- NULL

  # parse formulas
  if(verbose) cat("Preparing additive formula(s)...")
  parsed_formulas_contents <- lapply(1:length(list_of_formulas),
                                     function(i){

                                       if(!is.null(attr(additional_processors, "controls")))
                                         so <- c(so, attr(additional_processors, "controls"))
                                       if(!is.null(attr(list_of_formulas[[i]], "with_layer"))){
                                         so$with_layer <- attr(list_of_formulas[[i]], "with_layer")
                                       }else{
                                         so$with_layer <- TRUE
                                       }
                                       so$weight_options <- weight_options[[i]]

                                       res <- do.call("process_terms",
                                                      c(list(form = list_of_formulas[[i]],
                                                           data = data,
                                                           controls = so,
                                                           output_dim = output_dim,
                                                           param_nr = i,
                                                           parsing_options = formula_options,
                                                           specials_to_oz =
                                                             specials_to_oz,
                                                           automatic_oz_check =
                                                             automatic_oz_check,
                                                           identify_intercept =
                                                             orthog_options$identify_intercept
                                                           ),
                                                        list_of_deep_models,
                                                        additional_processors))

                                       return(res)
                                     })

  names(parsed_formulas_contents) <- names(list_of_formulas)

  if(verbose) cat(" Done.\n")

  if(return_prepoc)
    return(parsed_formulas_contents)

  if(!is.list(subnetwork_builder)){
    subnetwork_builder <- list(subnetwork_builder)[
      rep(1,length(parsed_formulas_contents))
      ]
  }else{
    if(length(parsed_formulas_contents) !=
       length(subnetwork_builder))
      stop("If subnetwork_builder is a list",
           ", it must be of the same size as the ",
           "list_of_formulas.")
  }

  if(verbose) cat("Preparing subnetworks...")

  # create gam data inputs
  if(!is.null(so$gamdata)){
    gaminputs <- makeInputs(so$gamdata$data_trafos, "gam_inp")
  }

  # create additive predictor per formula
  additive_predictors <- lapply(1:length(parsed_formulas_contents), function(i)
    subnetwork_builder[[i]](parsed_formulas_contents,
                            deep_top = orthog_options$deep_top,
                            orthog_fun = orthog_options$orthog_fun,
                            split_fun = orthog_options$split_fun,
                            shared_layers = weight_options[[i]]$shared_layers,
                            param_nr = i,
                            gaminputs = gaminputs
                            )
  )
  if(verbose) cat(" Done.\n")

  names(additive_predictors) <- names(list_of_formulas)
  if(!is.null(so$gamdata)){
    gaminputs <- list(gaminputs)
    names(gaminputs) <- "gaminputs"
    additive_predictors <- c(gaminputs, additive_predictors)
  }else{
    additive_predictors <- c(list(NULL), additive_predictors)
  }

  # initialize model
  if(verbose) cat("Building model...")
  model <- model_builder(list_pred_param = additive_predictors,
                         family = family,
                         output_dim = output_dim,
                         ...)
  if(verbose) cat(" Done.\n")

  ret <- list(model = model,
              init_params =
                list(
                  list_of_formulas = list_of_formulas,
                  gamdata = so$gamdata,
                  additive_predictors = additive_predictors,
                  parsed_formulas_contents = parsed_formulas_contents,
                  y = y,
                  ellipsis = list(...),
                  family = family,
                  penalty_options = penalty_options,
                  orthog_options = orthog_options,
                  image_var = image_var
                ),
              fit_fun = fitting_function)


  class(ret) <- "deepregression"

  return(ret)

}

#' @title Define Predictor of a Deep Distributional Regression Model
#'
#'
#' @param list_pred_param list of input-output(-lists) generated from
#' \code{subnetwork_init}
#' @param family see \code{?deepregression}; if NULL, concatenated
#' \code{list_pred_param} entries are returned (after applying mapping if provided)
#' @param output_dim dimension of the output
#' @param mapping a list of integers. The i-th list item defines which element
#' elements of \code{list_pred_param} are used for the i-th parameter.
#' For example, \code{mapping = list(1,2,1:2)} means that \code{list_pred_param[[1]]}
#' is used for the first distribution parameter, \code{list_pred_param[[2]]} for
#' the second distribution parameter and  \code{list_pred_param[[3]]} for both
#' distribution parameters (and then added once to \code{list_pred_param[[1]]} and
#' once to \code{list_pred_param[[2]]})
#' @param from_family_to_distfun function to create a \code{dist_fun} 
#' (see \code{?distfun_to_dist}) from the given character \code{family}
#' @param from_distfun_to_dist function creating a tfp distribution based on the
#' prediction tensors and \code{dist_fun}. See \code{?distfun_to_dist}
#' @param add_layer_shared_pred layer to extend shared layers defined in \code{mapping}
#' @param trafo_list a list of transformation function to convert the scale of the
#' additive predictors to the respective distribution parameter
#' @return a list with input tensors and output tensors that can be passed
#' to, e.g., \code{keras_model}
#'
#' @export
from_preds_to_dist <- function(
  list_pred_param,
  family = NULL,
  output_dim = 1L,
  mapping = NULL,
  from_family_to_distfun = make_tfd_dist,
  from_distfun_to_dist = distfun_to_dist,
  add_layer_shared_pred = function(x, units) layer_dense(x, units = units,
                                                         use_bias = FALSE),
  trafo_list = NULL
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
        if(lpp[[ind]]$shape[[2]] < len_map[ind]){
          # less units than needed => add layer and then split
          lpp[[ind]] <- tf$split(
            lpp[[ind]] %>% add_layer_shared_pred(units = len_map[ind]*output_dim),
            as.integer(len_map[ind]/output_dim),
            axis=1L
          )
        }else if(lpp[[ind]]$shape[[2]] == len_map[ind]){
          # units match number needed = just split
          lpp[[ind]] <- tf$split(
            lpp[[ind]],
            as.integer(len_map[ind]/output_dim),
            axis=1L
          )
        }else{
          # more units than needed
          stop("Node ", lpp[[ind]]$name, " has more units than defined by the mapping.\n",
               "  Does your deep neural network has the correct output dimensions?")
        }

      }

      lpp <- unlist(lpp, recursive = FALSE)
      mapping <- as.list(unlist(mapping))

    }

    # store names as they are obstructive later
    names_lpp <- names(lpp)
    lpp <- unname(lpp)

    for(i in 1:nr_params){
      list_pred_param[[i]] <- layer_add_identity(lpp[which(sapply(mapping, function(mp) i %in% mp))])
    }

    if(!is.null(names_lpp)) names(list_pred_param) <- names_lpp[1:nr_params]

  }else{

    nr_params <- length(list_pred_param)

  }

  # check family
  if(!is.null(family)){
    if(is.character(family)){
      if(family %in% c("betar", "gammar", "pareto_ls", "inverse_gamma_ls")){

        dist_fun <- family_trafo_funs_special(family)

      }else{

        dist_fun <- from_family_to_distfun(family, output_dim = output_dim,
                                           trafo_list = trafo_list)

      }
    }else{ # assuming that family is a dist_fun already

      dist_fun <- family

    }
  }else{

    return(layer_concatenate_identity(unname(list_pred_param)))

  }
  nrparams_dist <- attr(dist_fun, "nrparams_dist")
  if(is.null(nrparams_dist)) nrparams_dist <- nr_params

  if(nrparams_dist < nr_params)
  {
    warning("More formulas specified than parameters available.",
            " Will only use ", nrparams_dist, " formula(e).")
    nr_params <- nrparams_dist
    list_pred_param <- list_pred_param[1:nrparams_dist]
  }else if(nrparams_dist > nr_params){
    stop("Length of list_of_formula (", nr_params,
         ") does not match number of distribution parameters (",
         nrparams_dist, ").")
  }

  if(is.null(names(list_pred_param))){
    names(list_pred_param) <- names_families(family)
  }

  # concatenate predictors
  preds <- layer_concatenate_identity(unname(list_pred_param))

  # generate output
  out <- from_distfun_to_dist(dist_fun, preds)

  return(out)

}

#' @title Function to define output distribution based on dist_fun
#'
#' @param dist_fun a distribution function as defined by \code{make_tfd_dist}
#' @param preds tensors with predictions
#' @return a symbolic tfp distribution
#' @export
#'
distfun_to_dist <- function(dist_fun, preds)
{

  # tfprobability::layer_distribution_lambda(preds, make_distribution_fn = dist_fun)
  ret <- suppressMessages(try(tfp$layers$DistributionLambda(dist_fun)(preds), silent = TRUE))
  if(inherits(ret, "try-error")) ret <- tfp$layers$DistributionLambda(dist_fun)(preds)
  return(ret)

}

#' @title Compile a Deep Distributional Regression Model
#'
#'
#' @param list_pred_param list of input-output(-lists) generated from
#' \code{subnetwork_init}
#' @param weights vector of positive values; optional (default = 1 for all observations)
#' @param optimizer optimizer used. Per default Adam
#' @param model_fun which function to use for model building (default \code{keras_model})
#' @param monitor_metrics Further metrics to monitor
#' @param from_preds_to_output function taking the list_pred_param outputs
#' and transforms it into a single network output
#' @param loss the model's loss function; per default evaluated based on
#' the arguments \code{family} and \code{weights} using \code{from_dist_to_loss}
#' @param additional_penalty a penalty that is added to the negative log-likelihood;
#' must be a function of model$trainable_weights with suitable subsetting
#' @param ... arguments passed to \code{from_preds_to_output}
#' @return a list with input tensors and output tensors that can be passed
#' to, e.g., \code{keras_model}
#'
#' @examples
#' set.seed(24)
#' n <- 500
#' x <- runif(n) %>% as.matrix()
#' z <- runif(n) %>% as.matrix()
#'
#' y <- x - z
#' data <- data.frame(x = x, z = z, y = y)
#'
#' # change loss to mse and adapt
#' # \code{from_preds_to_output} to work
#' # only on the first output column
#' mod <- deepregression(
#'  y = y,
#'  data = data,
#'  list_of_formulas = list(loc = ~ 1 + x + z, scale = ~ 1),
#'  list_of_deep_models = NULL,
#'  family = "normal",
#'  from_preds_to_output = function(x, ...) x[[1]],
#'  loss = "mse"
#' )
#'
#'
#' @export
#'
#'
keras_dr <- function(
  list_pred_param,
  weights = NULL,
  optimizer = tf$keras$optimizers$Adam(),
  model_fun = keras_model,
  monitor_metrics = list(),
  from_preds_to_output = from_preds_to_dist,
  loss = from_dist_to_loss(family = list(...)$family,
                           weights = weights),
  additional_penalty = NULL,
  ...
)
{

  if(!is.null(list_pred_param[[1]])){
    inputs_gam <- unlist(list_pred_param[[1]])
  }else{
    inputs_gam <- NULL
  }
  list_pred_param <- list_pred_param[-1]
  # extract predictor inputs
  inputs <- lapply(list_pred_param, function(x) x[1:(length(x)-1)])
  inputs <- unname(unlist(inputs, recursive = TRUE))
  if(!is.null(inputs_gam)){
    inputs <- unname(c(inputs_gam, unlist(inputs)))
  }
  # extract predictor outputs
  outputs <- lapply(list_pred_param, function(x) x[[length(x)]])
  # define single output of network
  out <- from_preds_to_output(outputs, ...)
  # define model
  model <- model_fun(inputs = inputs,
                     outputs = out)
  # additional loss
  if(!is.null(additional_penalty)){

    add_loss <- function(x) additional_penalty(
      model$trainable_weights
    )
    model$add_loss(add_loss)

  }
  # allow for optimizer as a function of the model
  if(is.function(optimizer)){
    optimizer <- optimizer(model)
  }
  # compile model
  model %>% compile(optimizer = optimizer,
                    loss = loss,
                    metrics = monitor_metrics)

  return(model)

}

#' Function to transform a distritbution layer output into a loss function
#'
#' @param family see \code{?deepregression}
#' @param ind_fun function applied to the model output before calculating the
#' log-likelihood. Per default independence is assumed by applying \code{tfd_independent}.
#' @param weights sample weights
#'
#' @return loss function
#'
#'
#'
from_dist_to_loss <- function(
  family,
  ind_fun = function(x) tfd_independent(x),
  weights = NULL
){

  # define weights to be equal to 1 if not given
  if(is.null(weights)) weights <- 1

  # the negative log-likelihood is given by the negative weighted
  # log probability of the dist
  if(!is.character(family) || family!="pareto_ls"){
    negloglik <- function(y, dist)
      - weights * (dist %>% ind_fun() %>% tfd_log_prob(y))
  }else{
    negloglik <- function(y, dist)
      - weights * (dist %>% ind_fun() %>% tfd_log_prob(y + dist$scale))
  }

  return(negloglik)

}
