
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

    if (print_members)
      cat("Fitting member", iter, "...")

    st1 <- Sys.time()

    this_mod <- x$model

    x_train <- prepare_data(x$init_params$parsed_formulas_content,
                            gamdata = x$init_params$gamdata$data_trafos)

    # make callbacks
    this_callbacks <- callbacks

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
      ret$weighthistory <- get_weights(x$model)

    if (!is.null(save_fun))
      ret$save_fun_result <- save_fun(this_mod)

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

  set_weights(x$model, original_weights)

  return(invisible(ret))

}

get_ensemble_distribution <- function(object, data = NULL, topK = NULL, ...) {

  ens <- object$ensemble_results
  n_ensemble <- length(ens)
  original_weights <- get_weights(object$model)

  if (is.null(topK))
    topK <- n_ensemble

  if (topK != n_ensemble)
    stop("Not implemented yet.")

  if (is.null(ens[[1]]$weighthistory))
    stop("Weights were not saved. Consider running `ensemble` with `save_weights = TRUE`.")

  ens_weights <- lapply(ens, function(x) {
    x$weighthistory
  })

  dists <- lapply(ens_weights, function(x) {
    set_weights(object$model, x)
    get_distribution(object, data = data)
  })

  shp <- dists[[1]]$shape$as_list()
  probs <- k_constant(1 / topK, shape = c(shp, n_ensemble))
  dcat <- tfd_categorical(probs = probs)

  mix_dist <- tfd_mixture(dcat, dists)

  set_weights(object$model, original_weights)

  return(mix_dist)
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
