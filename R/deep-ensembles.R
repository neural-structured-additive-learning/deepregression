
#' Generic deep ensemble function
#'
#' @param x model to ensemble
#' @param ... further arguments passed to the class-specific function
#'
#' @export
ensemble <- function (x, ...) {
  UseMethod("cv", x)
}

#' Ensemblind deepregression models
#'
#' @param x object of class \code{"deepregression"}
#'
#' @return Ensemble
#' @export
ensemble.deepregression <- function(
  x,
  n_ensemble = 5,
  reinitialize = TRUE,
  verbose = FALSE,
  patience = 20,
  plot = TRUE,
  print_members = TRUE,
  stop_if_nan = TRUE,
  mylapply = lapply,
  save_weights = FALSE,
  callbacks = list(),
  save_fun = NULL,
  ...
)
{

  original_weights <- x$model$get_weights()

  res <- mylapply(1:n_ensemble, function(iter){

    # Randomly initialize weights
    if (reinitialize)
      reinit_weights(x)

    this_fold <- cv_folds[[iter]]

    if(print_members) cat("Fitting member ", iter, " ... ")
    st1 <- Sys.time()

    this_mod <- x$model

    x_train <- prepare_data(x$init_params$parsed_formulas_content,
                            gamdata = x$init_params$gamdata$data_trafos)

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
                        y = x$init_params$y,
                        callbacks = this_callbacks,
                        verbose = verbose,
                        view_metrics = FALSE
                   )
    )

    args <- append(args, x$init_params$ellipsis)

    ret <- do.call(x$fit_fun, args)

    if (save_weights)
      ret$weighthistory <- weighthistory$weights_last_layer

    if (!is.null(save_fun))
      ret$save_fun_result <- save_fun(this_mod)

    if(stop_if_nan && any(is.nan(ret$metrics$validloss)))
      stop("Fold ", iter, " with NaN's in ")

    this_mod$set_weights(old_weights)
    td <- Sys.time() - st1

    if (print_members)
      cat("\nDone in", as.numeric(td), "", attr(td, "units"), "\n")

    return(ret)

  })

  class(res) <- c("drEnsemble", "list")

  if (plot) try(plot_cv(res), silent = TRUE)

  x$model$set_weights(old_weights)

  invisible(return(res))

}

#' Re-intialize model weights
#' @param object model to re-initialize
#' @export
reinit_weights <- function(object) {
  UseMethod("reinit_weights")
}

#' Re-initialize deepregression weights
#' @return invisible \code{NULL}
#' @method reinit_weights deepregression
reinit_weights.deepregression <- function(object) {
  lapply(object$model$layers, function(x) {
    x$build(x$input_shape)
  })
  return(invisible(NULL))
}
