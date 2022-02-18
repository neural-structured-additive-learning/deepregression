
#' Generic deep ensemble function
#'
#' @param x model to ensemble
#' @param ... further arguments passed to the class-specific function
#'
#' @export
ensemble <- function (x, ...) {
  UseMethod("ensemble", x)
}

#' Ensemblind deepregression models
#'
#' @param x object of class \code{"deepregression"}
#'
#' @return Ensemble
#'
#' @method ensemble deepregression
#'
#' @export
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
  save_weights = FALSE,
  callbacks = list(),
  save_fun = NULL,
  ...
)
{

  original_weights <- x$model$get_weights()

  res <- mylapply(1:n_ensemble, function(iter) {

    # Randomly initialize weights
    if (reinitialize)
      x <- reinit_weights(x)
    else
      set_weights(x$model, original_weights)

    if(print_members) cat("Fitting member", iter, "...")
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
                        x = x_train,
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
      stop("Member ", iter, " with NaN's in validation loss")

    td <- Sys.time() - st1

    if (print_members)
      cat("\nDone in", as.numeric(td), "", attr(td, "units"), "\n")

    return(ret)

  })

  class(res) <- c("drEnsemble", "list")

  # if (plot) try(plot.drEnsemble(res), silent = TRUE)

  set_weights(x$model, original_weights)

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
    # x$build(x$input_shape)
    dtype <- x$dtype
    dshape <- try(x$kernel$shape, silent = TRUE)
    dweight <- try(x$kernel_initializer(shape = dshape, dtype = dtype), silent = TRUE)
    try(x$set_weights(weights = list(dweight)), silent = TRUE)
  })

  return(invisible(object))
}
