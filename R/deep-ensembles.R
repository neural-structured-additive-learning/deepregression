
#' Generic deep ensemble function
#'
#' @param x model to ensemble
#' @param ... further arguments passed to the class-specific function
#'
#' @export
#'
ensemble <- function (x, ...) {
  UseMethod("ensemble", x)
}

#' Ensembling deepregression models
#'
#' @param x object of class \code{"deepregression"} to ensemble
#' @param n_ensemble numeric; number of ensemble members to fit
#' @param reinitialize logical; if \code{TRUE} (default), model weights are
#'     initialized randomly prior to fitting each member. Fixed weights are
#'     not affected
#' @param save_weights whether to save final weights of each ensemble member;
#'     defaults to \code{TRUE}
#' @param print_members logical; print results for each member
#' @inheritParams cv.deepregression
#' @param seed seed for reproducibility
#' @param ... further arguments passed to \code{object$fit_fun}
#'
#' @return object of class \code{"drEnsemble"}, containing the original
#'     \code{"deepregression"} model together with a list of ensembling
#'     results (training history and, if \code{save_weights} is \code{TRUE},
#'     the trained weights of each ensemble member)
#'
#' @method ensemble deepregression
#' @export ensemble.deepregression
#' @export
#'
ensemble.deepregression <- function(
  x,
  n_ensemble = 5,
  reinitialize = TRUE,
  mylapply = lapply,
  verbose = FALSE,
  patience = 20,
  plot = TRUE,
  print_members = TRUE,
  stop_if_nan = TRUE,
  save_weights = TRUE,
  callbacks = list(),
  save_fun = NULL,
  seed = seq_len(n_ensemble),
  ...
)
{

  if(x$engine == "tf") {
    original_weights <- x$model$get_weights()
  } else {
      original_weights <- x %>% get_weights_torch()
      }


  res <- mylapply(1:n_ensemble, function(iter) {

    # Randomly initialize weights
    if (reinitialize){
      x <- reinit_weights(x, seed[iter])
    } else {
      if(x$engine == "tf") {
        set_weights(x$model, original_weights)
        } else {
          x$model()$load_state_dict(original_weights)
          }
    }

    if (print_members)
      cat("Fitting member", iter, "...")

    st1 <- Sys.time()

    if(x$engine == "tf") {
      this_mod <- x$model
    } else{
      this_mod <- x$model()
      }

    
    x_train <- prepare_data(x$init_params$parsed_formulas_content,
                            gamdata = x$init_params$gamdata$data_trafos, 
                            engine = x$engine)
    
    if(x$engine == "torch"){
    input_list_model <- 
      prepare_input_list_model(input_x = x_train,
                               input_y = as.matrix(x$init_params$y),
                               object = x,
                               callbacks = callbacks,
                               verbose = verbose,
                               view_metrics = F,
                               ...)
    }
    
    # make callbacks
    this_callbacks <- callbacks
    args <- list(...)
    
    if(x$engine == "tf"){
      args <- append(args,
                     list(object = this_mod,
                          x = x_train,
                          y = x$init_params$y,
                          callbacks = this_callbacks,
                          verbose = verbose,
                          view_metrics = FALSE
                     )
      )
    } else {
      args <- append(args,
                     input_list_model[!names(input_list_model) %in%
                                      names(args)])
    }
    
    args <- append(args, x$init_params$ellipsis)

    ret <- do.call(x$fit_fun, args)
  
  
    if (save_weights){
      if(x$engine == "tf") {
        ret$weighthistory <- get_weights(x$model)
      } else {
        ret$weighthistory <- x %>% get_weights_torch()
        }
    }

    if (!is.null(save_fun)){
      ret$save_fun_result <- save_fun(this_mod)
    }

    if(stop_if_nan && any(is.nan(ret$metrics$validloss)))
      stop("Member ", iter, " with NaN's in validation loss")

    td <- Sys.time() - st1

    if (print_members)
      cat("\nDone in", as.numeric(td), "", attr(td, "units"), "\n")

    return(ret)
  })

  ret <- c(x, ensemble_results = list(res))
  
  class(ret) <- c("drEnsemble", class(x))
  # if (plot) try(plot.drEnsemble(res), silent = TRUE)

  if(x$engine == "tf") {
    set_weights(x$model, original_weights)
  } else{
    x$model()$load_state_dict(original_weights)
  }


  return(invisible(ret))

}

#' Obtain the conditional ensemble distribution
#'
#' @param object object of class \code{"drEnsemble"}
#' @param data data for which to return the fitted distribution
#' @param topK not implemented yet
#' @param ... further arguments currently ignored
#'
#' @return \code{tfd_distribution} of the ensemble, i.e., a mixture of the
#'     ensemble member's predicted distributions conditional on \code{data}
#'
#' @export
#'
get_ensemble_distribution <- function(object, data = NULL, topK = NULL, ...) {

  ens <- object$ensemble_results
  n_ensemble <- length(ens)
  
  if(object$engine == "tf") {
    original_weights <- get_weights(object$model)
  } else {
    original_weights <- get_weights_torch(object)
    }
  
  if (is.null(topK))
    topK <- n_ensemble

  if (topK != n_ensemble)
    stop("Not implemented yet.")

  if (is.null(ens[[1]]$weighthistory))
    stop("Weights were not saved. Consider running `ensemble` with `save_weights = TRUE`.")

  dists <- .call_for_all_members(object, get_distribution, data = data)
  
  if(object$engine == "tf") {
    shp <- dists[[1]]$shape$as_list()
  } else {
    shp <- dists[[1]]$batch_shape[1]
    }
  
  if(object$engine == "tf") {
    probs <- k_constant(1 / topK,
                        shape = c(shp, n_ensemble))
  } else {
    probs <- torch::torch_full(size =  c(shp, n_ensemble),
                        fill_value = 1 / topK)
    }

  if(object$engine == "tf") {
    dcat <- tfd_categorical(probs = probs)
  } else {
    dcat <- torch::distr_categorical(probs = probs)
    }
  
  if(object$engine == "tf") {
    mix_dist <- tfd_mixture(dcat, dists)
  } else {
    used_distr <- family_to_trochd(family = object$init_params$family)
    distr_parameters <- prepare_torch_distr_mixdistr(object, dists)
    dists <- do.call(used_distr, distr_parameters)
    mix_dist <- torch::distr_mixture_same_family(dcat, dists)
    }
  


  if(object$engine == "tf") {
    set_weights(object$model, original_weights)
  } else {
    object$model()$load_state_dict(original_weights)
    }

  
  return(mix_dist)
}

#' Method for extracting ensemble coefficient estimates
#'
#' @param object object of class \code{"drEnsemble"}
#' @param ... further arguments supplied to \code{coef.deepregression}
#' @inheritParams coef.deepregression
#'
#' @return list of coefficient estimates of all ensemble members
#'
#' @method coef drEnsemble
#'
#' @export
#'
coef.drEnsemble <- function(object, which_param = 1, type = NULL, ...) {
  coefs <- .call_for_all_members(object, coef.deepregression,
                                 which_param = which_param, type = type,
                                 ... = ...)

  nms <- names(coefs[[1]])

  ret <- lapply(nms, function(nm) {
    do.call("cbind", lapply(coefs, function(member) {
      member[[nm]]
    }))
  })

  names(ret) <- nms

  ret

}

#' Method for extracting the fitted values of an ensemble
#'
#' @inheritParams fitted.deepregression
#'
#' @return list of fitted values for each ensemble member
#'
#' @method fitted drEnsemble
#'
#' @export
#'
fitted.drEnsemble <- function(object, apply_fun = tfd_mean, ...) {
  if(object$engine == "torch")  apply_fun = function(x) x$mean
  .call_for_all_members(object, fitted.deepregression,
                        apply_fun = apply_fun,
                        ... = ...)
}

.call_for_all_members <- function(object, FUN, ...) {
  ens_weights <- lapply(object$ensemble_results, function(x) {
    x$weighthistory
  })
  lapply(ens_weights, function(x) {
    if(object$engine == "tf") {
      set_weights(object$model, x)
    } else {
      object$model()$load_state_dict(x)
      } 
    FUN(object, ... = ...)
  })
}

#' Generic function to re-intialize model weights
#'
#' @param object model to re-initialize
#' @param seed seed for reproducibility
#' @export
#'
reinit_weights <- function(object, seed) {
  UseMethod("reinit_weights")
}

#' Method to re-initialize weights of a \code{"deepregression"} model
#'
#' @param object object of class \code{"deepregression"}
#' @param seed seed for reproducibility
#' @return invisible \code{NULL}
#'
#' @method reinit_weights deepregression
#'
#' @export
#'
reinit_weights.deepregression <- function(object, seed) {
  if(object$engine == "tf"){
  lapply(object$model$layers, function(x) {
    # x$build(x$input_shape)
    dtype <- x$dtype
    try({
      dshape <- x$kernel$shape
      dinit <- x$kernel_initializer$from_config(
        config = list(seed = tf$constant(seed)))
      dweight <- dinit(shape = dshape, dtype = dtype)
      x$set_weights(weights = list(dweight))
    }, silent = TRUE)
  })
  } else {
    torch::torch_manual_seed(seed)
    object$model()$apply(weight_reset)
  }
  return(invisible(object))
  
}
