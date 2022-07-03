#' @title Generic functions for deepregression models
#'
#' @param x deepregression object
#' @param which character vector or number(s) identifying the effect to plot; 
#' default plots all effects
#' @param which_param integer of length 1.
#' Corresponds to the distribution parameter for
#' which the effects should be plotted.
#' @param only_data logical, if TRUE, only the data for plotting is returned
#' @param grid_length the length of an equidistant grid at which a two-dimensional function
#' is evaluated for plotting.
#' @param main_multiple vector of strings; plot main titles if multiple plots are selected
#' @param type the type of plot (see generic \code{plot} function)
#' @param get_weight_fun function to extract weight from model given \code{x},
#' a \code{name} and \code{param_nr}
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
  only_data = FALSE,
  grid_length = 40,
  main_multiple = NULL,
  type = "b",
  get_weight_fun = get_weight_by_name,
  ... # passed to plot function
)
{

  pfc <- x$init_params$parsed_formulas_contents[[which_param]]
  plotable <- sapply(pfc, function(x) !is.null(x$plot_fun))
  names_all <- get_names_pfc(pfc)
  names <- names_all[plotable]
  
  if(!is.null(which)){
    if(!is.character(which)){
      which <- names[which]
    }else{
      which <- intersect(names, which)
    }
    if(length(which)==0)
      return("No smooth effects. Nothing to plot.")
  }else if(length(names)==0){
    return("No smooth effects. Nothing to plot.")
  }else{
    which <- names
  }
  
  plotData <- list()
    
  for(name in which){
    
    if(!is.null(main_multiple)){
      main <- main_multiple[1]
    }else if(!is.null(list(...)$main)){
      main <- list(...)$main
    }else{
      main <- name
    }
    
    pp <- pfc[[which(names_all==name)]]
    plotData[[name]] <- pp$plot_fun(pp, 
                                    weights = get_weight_fun(x, name = name , 
                                                             param_nr = which_param), 
                                    grid_length = grid_length)
    
    dims <- NCOL(plotData[[name]]$value)
    
    if(dims==1){
      
      if(!only_data){ 
       
        # necessary for multivariate outcomes
        for(i in 1:NCOL(plotData[[name]]$partial_effect)){ 
          
          suppressWarnings(plot(partial_effect[order(value),i] ~ sort(value),
                                data = plotData[[name]][c("value", "partial_effect")],
                                main = main,
                                xlab = extractvar(name),
                                ylab = "partial effect",
                                type = type,
                                ...))
          
        }
        
      }
      
    }else if(dims==2){
      
      if(!only_data){ 
        
        for(k in 1:NCOL(plotData[[name]]$partial_effect)){
          
          if(is.factor(plotData[[name]]$y)){
            
            ind <- rep(levels(plotData[[name]]$y), 
                       each = length(plotData[[name]]$x))
            
            for(lev in levels(plotData[[name]]$y))
              suppressWarnings(
                plot(
                  plotData[[name]]$x,
                  plotData[[name]]$partial_effect[ind==lev],
                  type = type,
                  xlab = colnames(plotData[[name]]$df)[1],
                  ylab = "partial effect",
                  main = gsub(colnames(plotData[[name]]$df)[2], lev, main),
                  ...
                )
              )
            
          }else{
            
            suppressWarnings(
              filled.contour(
                plotData[[name]]$x,
                plotData[[name]]$y,
                matrix(plotData[[name]]$partial_effect[,k], 
                       ncol=length(plotData[[name]]$y)),
                ...,
                xlab = colnames(plotData[[name]]$df)[1],
                ylab = colnames(plotData[[name]]$df)[2],
                # zlab = "partial effect",
                main = main
              )
            )
            
          }
          
        }
        
      }
      
    }else{
      warning("Plotting of effects with ", dims,
              " covariate inputs not supported.")
    }
    
    main_multiple <- main_multiple[-1]
    
  }
  
  invisible(plotData)
  
}


#' Predict based on a deepregression object
#'
#' @param object a deepregression model
#' @param newdata optional new data, either data.frame or list
#' @param batch_size batch_size for generator (image or large data use case)
#' @param apply_fun which function to apply to the predicted distribution,
#' per default \code{tfd_mean}, i.e., predict the mean of the distribution
#' @param convert_fun how should the resulting tensor be converted,
#' per default \code{as.matrix}
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
  ...
)
{

  # image case
  if(length(object$init_params$image_var)>0 | !is.null(batch_size)){
    
    return(predict_gen(object, newdata, batch_size, apply_fun, convert_fun))
  
  }else{
    
    if(is.null(newdata)){
      yhat <- object$model(prepare_data(object$init_params$parsed_formulas_contents,
                                        gamdata = object$init_params$gamdata$data_trafos))
    }else{
      # preprocess data
      if(is.data.frame(newdata)) newdata <- as.list(newdata)
      newdata_processed <- prepare_newdata(object$init_params$parsed_formulas_contents, 
                                           newdata, 
                                           gamdata = object$init_params$gamdata$data_trafos)
      yhat <- object$model(newdata_processed)
    }
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

#' Fit a deepregression model (pendant to fit for keras)
#'
#' @param object a deepregresison object.
#' @param batch_size integer, the batch size used for mini-batch training
#' @param epochs integer, the number of epochs to fit the model
#' @param early_stopping logical, whether early stopping should be user.
#' @param early_stopping_metric character, based on which metric should
#' early stopping be trigged (default: "val_loss")
#' @param verbose logical, whether to print losses during training.
#' @param view_metrics logical, whether to trigger the Viewer in RStudio / Browser.
#' @param patience integer, number of rounds after which early stopping is done.
#' @param save_weights logical, whether to save weights in each epoch.
#' @param validation_data optional specified validation data
#' @param validation_split float in [0,1] defining the amount of data used for validation
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
  batch_size = 32,
  epochs = 10,
  early_stopping = FALSE,
  early_stopping_metric = "val_loss",
  verbose = TRUE,
  view_metrics = FALSE,
  patience = 20,
  save_weights = FALSE,
  validation_data = NULL,
  validation_split = ifelse(is.null(validation_data), 0.1, 0),
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
                        list(callback_terminate_on_naan(),
                             callback_early_stopping(patience = patience,
                                                     restore_best_weights = TRUE,
                                                     monitor = early_stopping_metric)
                             )
    )
  
  args <- list(...)

  input_x <- prepare_data(object$init_params$parsed_formulas_content, 
                          gamdata = object$init_params$gamdata$data_trafos)
  input_y <- as.matrix(object$init_params$y)
  
  if(!is.null(validation_data))
    validation_data <- 
    list(
      x = prepare_newdata(object$init_params$parsed_formulas_content, 
                          validation_data[[1]], 
                          gamdata = object$init_params$gamdata$data_trafos),
      y = as.matrix(validation_data[[2]], ncol=1)
    )

  if(length(object$init_params$image_var)>0){
    
    args <- prepare_generator_deepregression(
      x = object$model,
      input_x = input_x,
      input_y = input_y,
      sizes = object$init_params$image_var,
      validation_data = validation_data,
      batch_size = batch_size,
      epochs = epochs,
      verbose = verbose,
      validation_split = validation_split,
      callbacks = callbacks,
      ...
    )

  }else{
    
    input_list_model <-
      list(object = object$model,
           epochs = epochs,
           batch_size = batch_size,
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
    
  }

  ret <- suppressWarnings(do.call(object$fit_fun, args))
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
#' @param ... not used
#'
#' @importFrom stats coef
#' @method coef deepregression
#' @rdname methodDR
#' @export
#'
coef.deepregression <- function(
  object,
  which_param = 1,
  type = NULL,
  ...
)
{
  
  pfc <- object$init_params$parsed_formulas_contents[[which_param]]
  to_return <- get_type_pfc(pfc, type)
  
  names <- get_names_pfc(pfc)[as.logical(to_return)]
  if(any(grepl("^mult\\(", names))){
    names_mult <- c(sapply(names[grepl("^mult\\(", names)], get_terms_mult))
    names <- c(names[!grepl("^mult\\(", names)], names_mult)
  }
  pfc <- pfc[as.logical(to_return)]
  check_names <- names
  check_names[check_names=="(Intercept)"] <- "1"
  
  coefs <- lapply(1:length(check_names), function(i) 
    pfc[[i]]$coef(get_weight_by_name(object, check_names[i], which_param)))
  
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

#' Generic cv function
#'
#' @param x model to do cv on
#' @param ... further arguments passed to the class-specific function
#'
#' @export
cv <- function (x, ...) {
  UseMethod("cv", x)
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
#' @param cv_folds an integer; can also be a list of lists 
#' with train and test data sets per fold
#' @param stop_if_nan logical; whether to stop CV if NaN values occur
#' @param callbacks a list of callbacks used for fitting
#' @param save_fun function applied to the model in each fold to be stored in
#' the final result
#' @export
#' @rdname methodDR
#'
#' @return Returns an object \code{drCV}, a list, one list element for each fold
#' containing the model fit and the \code{weighthistory}.
#'
#'
#'
cv.deepregression <- function(
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
  save_fun = NULL,
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

  res <- mylapply(1:length(cv_folds), function(folds_iter){

    this_fold <- cv_folds[[folds_iter]]
    
    if(print_folds) cat("Fitting Fold ", folds_iter, " ... ")
    st1 <- Sys.time()

    # does not work?
    # this_mod <- clone_model(x$model)
    this_mod <- x$model

    train_ind <- this_fold[[1]]
    test_ind <- this_fold[[2]]

    x_train <- prepare_data(x$init_params$parsed_formulas_content,
                            gamdata = x$init_params$gamdata$data_trafos)
    
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
    if(!is.null(save_fun))
      ret$save_fun_result <- save_fun(this_mod)
    
    if(stop_if_nan && any(is.nan(ret$metrics$validloss)))
      stop("Fold ", folds_iter, " with NaN's in ")

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
stddev <- function (x, ...) {
  UseMethod("stddev", x)
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
stddev.deepregression <- function(
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
quant <- function (x, ...) {
  UseMethod("quant", x)
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
quant.deepregression <- function(
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
    disthat <- x$model(prepare_data(x$init_params$parsed_formulas_content, 
                                    gamdata = x$init_params$gamdata$data_trafos))
  }else{
    # preprocess data
    if(is.data.frame(data)) data <- as.list(data)
    newdata_processed <- prepare_newdata(x$init_params$parsed_formulas_content, 
                                         data, 
                                         gamdata = x$init_params$gamdata$data_trafos)
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
  ind_fun = function(x) tfd_independent(x),
  convert_fun = as.matrix,
  summary_fun = function(x) x
)
{

  if(is.null(data)){
    
    this_data <- prepare_data(x$init_params$parsed_formulas_content, 
                              gamdata = x$init_params$gamdata$data_trafos)
  
  }else{
    
    if(is.data.frame(data)) data <- as.list(data)
    this_data <- prepare_newdata(x$init_params$parsed_formulas_content, 
                                 data, 
                                 gamdata = x$init_params$gamdata$data_trafos)
    
  }
  
  disthat <- x$model(this_data)
    
  if(is.null(this_y)){
    this_y <- x$init_params$y
  }else{
    if(is.null(dim(this_y)))
      warning("log-score calculation requires this_y to be a matrix.")
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
#' @param postfixes character (vector) appended to layer name
#' @return weight matrix
#' 
#' 
get_weight_by_name <- function(mod, name, param_nr=1, postfixes="")
{

  # check for shared layer  
  names_pfc <- get_names_pfc(mod$init_params$parsed_formulas_contents[[param_nr]])
  names_pfc[names_pfc=="(Intercept)"] <- "1"
  pfc_term <- mod$init_params$parsed_formulas_contents[[param_nr]][[which(names_pfc==name)]]
  if(!is.null(pfc_term$shared_name)){
    this_name <- paste0(pfc_term$shared_name, postfixes)
  }else{
    this_name <- paste0(makelayername(name, param_nr), postfixes)
  }
  # names <- get_mod_names(mod)
  if(length(this_name)>1){
    wgts <- lapply(this_name, function(name) get_weight_by_opname(mod, name))
  }else{
    wgts <- get_weight_by_opname(mod, this_name)
  }
  return(wgts)
  
}


#' Return partial effect of one smooth term
#' 
#' @param object deepregression object
#' @param names string; for partial match with smooth term
#' @param return_matrix logical; whether to return the design matrix or
#' @param which_param integer; which distribution parameter
#' the partial effect (\code{FALSE}, default)
#' @param newdata data.frame; new data (optional)
#' @param ... arguments passed to \code{get_weight_by_name}
#' 
#' @export
#' 
get_partial_effect <- function(object, names=NULL, return_matrix = FALSE, 
                               which_param = 1, newdata = NULL, ...)
{
  
  names_pfc <- get_names_mod(object, which_param)
  names <- if(!is.null(names)) intersect(names, names_pfc) else names_pfc
  
  if(length(names)==0)
    stop("Cannot find specified name(s) in additive predictor #", which_param,".")
  
  res <- lapply(names, function(name){
    w <- which(name==names_pfc)
    
    if(name=="(Intercept)") name <- "1"
    weights <- get_weight_by_name(object, name = name, param_nr = which_param, ...)
    weights <- object$init_params$parsed_formulas_contents[[which_param]][[w]]$coef(weights)
    pe_fun <- object$init_params$parsed_formulas_contents[[which_param]][[w]]$partial_effect
    if(is.null(pe_fun)){
      #warning("Specified term does not have a partial effect function. Returning weights.")
      return(weights)
    }else{
      return(pe_fun(weights, newdata))
    }
  })
  if(length(res)==1) return(res[[1]]) else return(res)
  
}

