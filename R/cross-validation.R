make_cv_list_simple <- function(data_size, folds, seed = 42, shuffle = TRUE)
{

  set.seed(seed)
  suppressWarnings(
    mysplit <- split(sample(1:data_size),
                     f = rep(1:folds, each = data_size/folds))
  )
  lapply(mysplit, function(test_ind)
    list(train_ind = setdiff(1:data_size, test_ind),
         test_ind = test_ind))

}

extract_cv_result <- function(res, name_loss = "loss", name_val_loss = "val_loss"){

  losses <- sapply(res, "[[", "metrics")
  trainloss <- data.frame(losses[name_loss,])
  validloss <- data.frame(losses[name_val_loss,])
  weightshist <- lapply(res, "[[", "weighthistory")

  return(list(trainloss=trainloss,
              validloss=validloss,
              weight=weightshist))

}

#' Plot CV results from deepregression
#'
#' @param x \code{drCV} object returned by \code{cv.deepregression}
#' @param what character indicating what to plot (currently supported 'loss'
#' or 'weights')
#' @param ... further arguments passed to \code{matplot}
#'
#' @export
#'
plot_cv <- function(x, what=c("loss","weight"), ...){

  .pardefault <- par(no.readonly = TRUE)
  cres <- extract_cv_result(x)

  what <- match.arg(what)

  if(what=="loss"){

    loss <- cres$trainloss
    mean_loss <- apply(loss, 1, mean)
    vloss <- cres$validloss
    mean_vloss <- apply(vloss, 1, mean)

    oldpar <- par(no.readonly = TRUE)    # code line i
    on.exit(par(oldpar))            # code line i + 1
    par(mfrow=c(1,2))
    matplot(loss, type="l", col="black", ..., ylab="loss", xlab="epoch")
    points(1:(nrow(loss)), mean_loss, type="l", col="red", lwd=2)
    abline(v=which.min(mean_loss), lty=2)
    matplot(vloss, type="l", col="black", ...,
            ylab="validation loss", xlab="epoch")
    points(1:(nrow(vloss)), mean_vloss, type="l", col="red", lwd=2)
    abline(v=which.min(mean_vloss), lty=2)
    suppressWarnings(par(.pardefault))

  }else{

    stop("Not implemented yet.")

  }

  invisible(NULL)

}

#' Function to get the stoppting iteration from CV
#' @param res result of cv call
#' @param thisFUN aggregating function applied over folds
#' @param loss which loss to use for decision
#' @param whichFUN which function to use for decision
#'
#' @export
stop_iter_cv_result <- function(res, thisFUN = mean,
                                loss = "validloss",
                                whichFUN = which.min)
{

  whichFUN(apply(extract_cv_result(res)[[loss]], 1, FUN=thisFUN))

}

#' Generate folds for CV out of one hot encoded matrix
#'
#' @param mat matrix with columns corresponding to folds
#' and entries corresponding to a one hot encoding
#' @param val_train the value corresponding to train, per default 0
#' @param val_test the value corresponding to test, per default 1
#'
#' @details
#' \code{val_train} and \code{val_test} can both be a set of value
#'
#' @export
make_folds <- function(mat, val_train=0, val_test=1)
{

  apply(mat, 2, function(x){
    list(train = which(x %in% val_train),
         test = which(x %in% val_test))
  })

}
