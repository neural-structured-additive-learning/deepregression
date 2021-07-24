#' @title Generic functions for deepregression models
#'
#' @param x deepregression object
#' @param name which effect to plot as character; default plots all
#' @param which_param integer of length 1.
#' Corresponds to the distribution parameter for
#' which the effects should be plotted.
#' @param plot logical, if FALSE, only the data for plotting is returned
#' @param grid_length the length of an equidistant grid at which a two-dimensional function
#' is evaluated for plotting.
#' @param ... further arguments, passed to fit, plot or predict function
#'
#' @method plot deepregression
#' @export
#' @rdname methodDR
#'
plot.deepregression <- function(
  x,
  which = NULL,
  # which of the nonlinear structured effects
  which_param = 1, # for which parameter
  plot = TRUE,
  grid_length = 40,
  ... # passed to plot function
)
{

  pfc <- x$init_params$parsed_formulas_contents[[which_param]]
  plotable <- sapply(pfc, function(x) !is.null(x$plot_fun))
  names_all <- get_names_pfc(pfc)
  names <- names_all[plotable]
  
  if(!is.null(which)){
    which <- intersect(names, which)
    if(length(which)==0)
      return("No smooth effects. Nothing to plot.")
  }else{
    which <- names
  }
  
  plotData <- list()
    
  for(name in which){
    
    pp <- pfc[[which(names_all==name)]]
    plotData[[name]] <- pp$plot_fun(pp, 
                                    weights = get_weight_by_name(x, name = name , 
                                                                 param_nr = which_param), 
                                    grid_length = grid_length)
    
    dims <- NCOL(plotData[[name]]$value)
    
    if(dims==1){
      
      if(plot) suppressWarnings(plot(partial_effect[order(value),1] ~ sort(value),
                    data = plotData[[name]][c("value", "partial_effect")],
                    main = name,
                    xlab = extractvar(name),
                    ylab = "partial effect",
                    ...))
      
    }else if(dims==2){
      
      if(plot) suppressWarnings(
        filled.contour(
          plotData[[name]]$x,
          plotData[[name]]$y,
          matrix(plotData[[name]]$partial_effect, 
                 ncol=length(plotData[[name]]$y)),
          ...,
          xlab = colnames(plotData[[name]]$df)[1],
          ylab = colnames(plotData[[name]]$df)[2],
          # zlab = "partial effect",
          main = name
        )
      )
      
    }else{
      warning("Plotting of effects with ", dims,
              " covariate inputs not supported.")
    }
    
  }
  
  invisible(plotData)
  
}


#' Predict based on a deepregression object
#'
#' @param object a deepregression model
#' @param newdata optional new data, either data.frame or list
#' @param batch_size batch_size for generator (image use cases)
#' @param apply_fun which function to apply to the predicted distribution,
#' per default \code{tfd_mean}, i.e., predict the mean of the distribution
#' @param convert_fun how should the resulting tensor be converted,
#' per default \code{as.matrix}
#' @param dtype string for conversion 
#'
#' @export
#' @rdname methodDR
#'
predict.deepregression <- function(
  object,
  newdata = NULL,
  batch_size = NULL,
  apply_fun = tfd_mean,
  convert_fun = as.matrix,
  dtype = "float32",
  ...
)
{

    if(is.null(newdata)){
      yhat <- object$model(prepare_data(object$init_params$parsed_formulas_contents))
    }else{
      # preprocess data
      if(is.data.frame(newdata)) newdata <- as.list(newdata)
      newdata_processed <- prepare_newdata(object$init_params$parsed_formulas_contents, 
                                           newdata)
      yhat <- object$model(newdata_processed)
    }
   
    if(!is.null(apply_fun))
      return(convert_fun(apply_fun(yhat))) else
        return(convert_fun(yhat))

}

#' Function to extract fitted distribution
#'
#' @param object a deepregression object
#' @param apply_fun function applied to fitted distribution,
#' per default \code{tfd_mean}
#' @param ... further arguments passed to the predict function
#'
#' @export
#' @rdname methodDR
#'
fitted.deepregression <- function(
  object, apply_fun = tfd_mean, ...
)
{
  return(
    predict.deepregression(object, apply_fun=apply_fun, ...)
  )
}

#' Generic train function
#'
#' @param x object
#' @param ... further arguments passed to the class-specific function
#'
#' @export
fit <- function (object, ...) {
  UseMethod("fit", object)
}

#' Fit a deepregression model (pendant to fit for keras)
#'
#' @param object a deepregresison object.
#' @param batch_size 
#' @param early_stopping logical, whether early stopping should be user.
#' @param verbose logical, whether to print losses during training.
#' @param view_metrics logical, whether to trigger the Viewer in RStudio / Browser.
#' @param patience integer, number of rounds after which early stopping is done.
#' @param save_weights logical, whether to save weights in each epoch.
#' @param auc_callback logical, whether to use a callback for AUC
#' @param validation_data optional specified validation data
#' @param callbacks a list of callbacks for fitting
#' @param convertfun function to convert R into Tensor object
#' @param ... further arguments passed to
#' \code{keras:::fit.keras.engine.training.Model}
#'
#'
#' @export fit deepregression
#' @export
#' 
#' @rdname methodDR
#'
fit.deepregression <- function(
  object,
  batch_size = NULL,
  epochs = 10,
  early_stopping = FALSE,
  verbose = TRUE,
  view_metrics = FALSE,
  patience = 20,
  save_weights = FALSE,
  validation_data = NULL,
  validation_split = 0.1,
  callbacks = list(),
  convertfun = function(x) tf$constant(x, dtype="float32"),
  ...
)
{

  # make callbacks
  if(save_weights){
    weighthistory <- WeightHistory$new()
    callbacks <- append(callbacks, weighthistory)
  }
  if(early_stopping & length(callbacks)==0)
    callbacks <- append(callbacks,
                        callback_early_stopping(patience = patience))
  
  args <- list(...)

  input_x <- prepare_data(object$init_params$parsed_formulas_content)
  input_y <- as.matrix(object$init_params$y)
  
  if(!is.null(validation_data))
    validation_data <- 
    list(
      x = prepare_newdata(object$init_params$parsed_formulas_content, validation_data[[1]]),
      y = as.matrix(validation_data[[2]], ncol=1)
    )

  condition_for_images <- FALSE
  if(condition_for_images){
    ret <- fit_generator_deepregression
    return(invisible(ret))
  }


  
  input_list_model <-
    list(object = object$model,
         validation_split = validation_split,
         validation_data = validation_data,
         callbacks = callbacks,
         verbose = verbose,
         view_metrics = ifelse(view_metrics, getOption("keras.view_metrics", default = "auto"), FALSE)
    )
  
  input_list_model <- c(input_list_model,
                        list(x = input_x,
                             y = input_y
                        ))
 
  args <- append(args,
                 input_list_model[!names(input_list_model) %in%
                                    names(args)])

  ret <- do.call(object$fit_fun, args)
  if(save_weights) ret$weighthistory <- weighthistory$weights_last_layer
  invisible(ret)
}

#' Extract layer weights / coefficients from model
#'
#' @param object a deepregression model
#' @param which_param integer, indicating for which distribution parameter
#' coefficients should be returned (default is first parameter)
#' @param type either NULL (all types of coefficients are returned),
#' "linear" for linear coefficients or "smooth" for coefficients of 
#' smooth terms
#'
#' @method coef deepregression
#' @export
#' @rdname methodDR
#'
coef.deepregression <- function(
  object,
  which_param = 1,
  type = NULL
)
{
  
  pfc <- object$init_params$parsed_formulas_contents[[which_param]]
  linear <- sapply(pfc, function(x) is.null(x$partial_effect) & !is.null(x$coef))
  smooth <- sapply(pfc, function(x) !is.null(x$partial_effect) & !is.null(x$coef))
  
  if(is.null(type)) type <- c("linear", "smooth") else 
    stopifnot(type %in% c("linear", "smooth"))
  to_return <- linear * ("linear" %in% type) + smooth * ("smooth" %in% type)
  
  names <- get_names_pfc(pfc)[as.logical(to_return)]
  check_names <- names
  check_names[check_names=="(Intercept)"] <- "1"
  
  coefs <- lapply(check_names, function(n) get_weight_by_name(object, n, which_param))
  names(coefs) <- names
  
  return(coefs)

}

#' Print function for deepregression model
#'
#' @export
#' @rdname methodDR
#' @param x a \code{deepregression} model
#' @param ... unused
#'
#' @method print deepregression
#'
print.deepregression <- function(
  x,
  ...
)
{
  print(x$model)
  fae <- x$init_params$list_of_formulas
  cat("Model formulas:\n---------------\n")
  invisible(sapply(1:length(fae), function(i){ cat(names(fae)[i],":\n"); print(fae[[i]])}))
}

#' @title Cross-validation for deepgression objects
#' @param ... further arguments passed to
#' \code{keras:::fit.keras.engine.training.Model}
#' @param x deepregression object
#' @param verbose whether to print training in each fold
#' @param patience number of patience for early stopping
#' @param plot whether to plot the resulting losses in each fold
#' @param print_folds whether to print the current fold
#' @param mylapply lapply function to be used; defaults to \code{lapply}
#' @param save_weights logical, whether to save weights in each epoch.
#' @param cv_folds an integer if list with train and test data sets
#' @param stop_if_nan logical; whether to stop CV if NaN values occur
#' @param callbacks a list of callbacks used for fitting
#' @export
#' @rdname methodDR
#'
#' @return Returns an object \code{drCV}, a list, one list element for each fold
#' containing the model fit and the \code{weighthistory}.
#'
#'
#'
cv <- function(
  x,
  verbose = FALSE,
  patience = 20,
  plot = TRUE,
  print_folds = TRUE,
  cv_folds = 5,
  stop_if_nan = TRUE,
  mylapply = lapply,
  save_weights = FALSE,
  callbacks = list(),
  ...
)
{

  if(!is.list(cv_folds) & is.numeric(cv_folds)){
    cv_folds <- make_cv_list_simple(
      data_size = NROW(x$init_params$y),
      cv_folds)
  }
  
  nrfolds <- length(cv_folds)
  old_weights <- x$model$get_weights()

  if(print_folds) folds_iter <- 1

  # subset fun
  if(NCOL(x$init_params$y)==1)
    subset_fun <- function(y,ind) y[ind] else
      subset_fun <- function(y,ind) subset_array(y,ind)

  res <- mylapply(cv_folds, function(this_fold){

    if(print_folds) cat("Fitting Fold ", folds_iter, " ... ")
    st1 <- Sys.time()

    # does not work?
    # this_mod <- clone_model(x$model)
    this_mod <- x$model

    train_ind <- this_fold[[1]]
    test_ind <- this_fold[[2]]

    x_train <- prepare_data(x$init_params$parsed_formulas_content)
    
    train_data <- lapply(x_train, function(x)
        subset_array(x, train_ind))
    test_data <- lapply(x_train, function(x)
        subset_array(x, test_ind))
    
    # make callbacks
    this_callbacks <- callbacks
    if(save_weights){
      weighthistory <- WeightHistory$new()
      this_callbacks <- append(this_callbacks, weighthistory)
    }

    args <- list(...)
    args <- append(args,
                   list(object = this_mod,
                        x = train_data,
                        y = subset_fun(x$init_params$y, train_ind),
                        validation_split = NULL,
                        validation_data = list(
                          test_data,
                          subset_fun(x$init_params$y,test_ind)
                        ),
                        callbacks = this_callbacks,
                        verbose = verbose,
                        view_metrics = FALSE
                   )
    )
    
    args <- append(args, x$init_params$ellipsis)

    ret <- do.call(x$fit_fun, args)
    if(save_weights) ret$weighthistory <- weighthistory$weights_last_layer

    if(stop_if_nan && any(is.nan(ret$metrics$validloss)))
      stop("Fold ", folds_iter, " with NaN's in ")

    if(print_folds) folds_iter <<- folds_iter + 1

    this_mod$set_weights(old_weights)
    td <- Sys.time()-st1
    if(print_folds) cat("\nDone in", as.numeric(td), "", attr(td,"units"), "\n")

    return(ret)

  })

  class(res) <- c("drCV","list")

  if(plot) try(plot_cv(res), silent = TRUE)

  x$model$set_weights(old_weights)

  invisible(return(res))

}

#' mean of model fit
#'
#' @export
#' @rdname methodDR
#'
#' @param x a deepregression model
#' @param data optional data, a data.frame or list
#' @param ... arguments passed to the predict function
#'
#' @method mean deepregression
#'
#'
mean.deepregression <- function(
  x,
  data = NULL,
  ...
)
{
  predict.deepregression(x, newdata = data, apply_fun = tfd_mean, ...)
}


#' Generic sd function
#'
#' @param x object
#' @param ... further arguments passed to the class-specific function
#'
#' @export
sd <- function (x, ...) {
  UseMethod("sd", x)
}

#' Standard deviation of fit distribution
#'
#' @param x a deepregression object
#' @param data either NULL or a new data set
#' @param ... arguments passed to the \code{predict} function
#'
#' @export
#' @rdname methodDR
#'
sd.deepregression <- function(
  x,
  data = NULL,
  ...
)
{
  predict.deepregression(x, newdata = data, apply_fun = tfd_stddev, ...)
}

#' Generic quantile function
#'
#' @param x object
#' @param ... further arguments passed to the class-specific function
#'
#' @export
quantile <- function (x, ...) {
  UseMethod("quantile", x)
}

#' Calculate the distribution quantiles
#'
#' @param x a deepregression object
#' @param data either \code{NULL} or a new data set
#' @param probs the quantile value(s)
#' @param ... arguments passed to the \code{predict} function
#'
#' @export
#' @rdname methodDR
#'
quantile.deepregression <- function(
  x,
  data = NULL,
  probs,
  ...
)
{
  predict.deepregression(x,
                         newdata = data,
                         apply_fun = function(x) tfd_quantile(x, value=probs),
                         ...)
}

#' Function to return the fitted distribution
#'
#' @param x the fitted deepregression object
#' @param data an optional data set
#' @param force_float forces conversion into float tensors
#'
#' @export
#'
get_distribution <- function(
  x,
  data = NULL,
  force_float = FALSE
)
{
  if(is.null(data)){
    disthat <- x$model(prepare_data(x$init_params$parsed_formulas_content))
  }else{
    # preprocess data
    if(is.data.frame(data)) data <- as.list(data)
    newdata_processed <- prepare_newdata(x$init_params$parsed_formulas_content, 
                                         data)
    disthat <- x$model(newdata_processed)
  }
  return(disthat)
}

#' Function to return the log_score
#'
#' @param x the fitted deepregression object
#' @param data an optional data set
#' @param this_y new y for optional data
#' @param ind_fun function indicating the dependency; per default (iid assumption)
#' \code{tfd_independent} is used.
#' @param convert_fun function that converts Tensor; per default \code{as.matrix}
#' @param summary_fun function summarizing the output; per default the identity
#'
#' @export
log_score <- function(
  x,
  data=NULL,
  this_y=NULL,
  ind_fun = function(x) tfd_independent(x,1),
  convert_fun = as.matrix,
  summary_fun = function(x) x
)
{

  if(is.null(data)){
    
    this_data <- prepare_data(x$init_params$parsed_formulas_content)
  
  }else{
    
    if(is.data.frame(data)) data <- as.list(data)
    this_data <- prepare_newdata(x$init_params$parsed_formulas_content, 
                                 data)
    
  }
  
  disthat <- x$model(this_data)
    
  if(is.null(this_y)){
    this_y <- x$init_params$y
  }
  
  return(summary_fun(convert_fun(
    disthat %>% ind_fun() %>% tfd_log_prob(this_y)
  )))
}

#' Function to retrieve the weights of a structured layer
#' 
#' @param mod fitted deepregression object
#' @param name name of partial effect
#' @param param_nr distribution parameter number
#' @return weight matrix
#' 
#' 
get_weight_by_name <- function(mod, name, param_nr=1)
{
  
  name <- makelayername(name, param_nr)
  names <- sapply(mod$model$layers,"[[","name")
  w <- which(name==names)
  if(length(w)==0)
    stop("Cannot find specified name in additive predictor #", param_nr,".")
  wgts <- mod$model$layers[[w]]$weights
  if(is.list(wgts) & length(wgts)==1)
    return(as.matrix(wgts[[1]]))
  return(wgts)
  
}


#' Return partial effect of one smooth term
#' 
#' @param object deepregression object
#' @param name string; for partial match with smooth term
#' @param return_matrix logical; whether to return the design matrix or
#' @param which_param integer; which distribution parameter
#' the partial effect (\code{FALSE}, default)
#' @param newdata data.frame; new data (optional)
#' 
#' @export
#' 
get_partial_effect <- function(object, name, return_matrix = FALSE, 
                               which_param = 1, newdata = NULL)
{
  
  weights <- get_weight_by_name(object, name = name, param_nr = which_param)
  names_pfc <- get_names_pfc(object$init_params$parsed_formulas_contents[[which_param]])
  w <- which(name==names_pfc)
  if(length(w)==0)
    stop("Cannot find specified name in additive predictor #", which_param,".")
  pe_fun <- object$init_params$parsed_formulas_contents[[which_param]][[w]]$partial_effect
  if(is.null(pe_fun)){
    warning("Specified term does not have a partial effect function. Returning weights.")
    return(weights)
  }
  return(pe_fun(weights, newdata))
  
}

