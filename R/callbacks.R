WeightHistory <- R6::R6Class("WeightHistory",
                             inherit = KerasCallback,

                             public = list(

                               weights_last_layer = NULL,

                               on_epoch_end = function(batch, logs = list()) {
                                 self$weights_last_layer <-
                                   cbind(self$weights_last_layer,
                                         coefkeras(self$model))
                               }
                             ))


# define AUC callback
# https://github.com/rstudio/keras/issues/319
auc_roc <- R6::R6Class("AUC",
                       inherit = KerasCallback,
                       public = list(

                         losses = NULL,
                         x = NA,
                         y = NA,
                         x_val = NA,
                         y_val = NA,

                         initialize = function(training = list(), validation= list()){
                           self$x <- training[[1]]
                           self$y <- training[[2]]
                           self$x_val <- validation[[1]]
                           self$y_val <- validation[[2]]
                         },


                         on_epoch_end = function(epoch, logs = list()){

                           self$losses <- c(self$losses, logs[["loss"]])
                           y_pred <- as.matrix(tfd_mean(self$model(self$x)))
                           y_pred_val <- as.matrix(tfd_mean(self$model(self$x_val)))
                           score = auc(actual = c(self$y),
                                                predicted =  c(y_pred))
                           score_val = auc(actual = c(self$y_val),
                                                    predicted =  c(y_pred_val))
                           cat("epoch: ", epoch+1, " AUC:", round(score,6), ' AUC_val:',
                               round(score_val,6), "\n")
                         }
                       ))

################################################################################################

auc_metric <- custom_metric(name = "auc", metric_fn = function(y_true, y_pred) {

  auc.numpy.out <- function(y_true, y_pred){
    out <- Metrics::auc(predicted = y_pred, actual = y_true)
    return(tf$constant(out, "double"))
  }

  return(tf$numpy_function(func = auc.numpy.out,
                           inp = c(y_true,y_pred),
                           Tout = tf$double))
})
################################################################################################

tf_crps_norm <- function(h)
{
  return(
    h * (2 * tfd_normal(0,1)$cdf(h) - 1) + 
      (tf$math$sqrt(2) * tf$math$exp(-0.5 * tf$square(h)) - 1)/tf$math$sqrt(pi)
  )
}

crps_stdnorm_metric <- custom_metric(name = "crps_stdnorm", metric_fn = function(y_true, y_pred) {
  
  h_y_eval <- y_pred[,1, drop = FALSE] + y_pred[,2, drop = FALSE]
  
  return(tf_crps_norm(h_y_eval))
  
})

################################################################################################

# overwrite keras:::KerasMetricsCallback class
KerasMetricsCallback_custom <-
  R6::R6Class("KerasMetricsCallback",

              inherit = KerasCallback,

              public = list(

                # instance data
                metrics = list(),
                metrics_viewer = NULL,
                weights_last_layer = NULL,
                view_metrics = FALSE,

                initialize = function(view_metrics = FALSE) {
                  self$view_metrics <- view_metrics
                },

                on_train_begin = function(logs = NULL) {

                  # add weight_diff to metrics
                  self$params$metrics <- c(self$params$metrics,
                                           "weight_diff")
                  # strip validation metrics if do_validation is FALSE (for
                  # fit_generator and fitting TF record the val_ metrics are
                  # passed even though no data will be provided for them)
                  if (!self$params$do_validation) {
                    self$params$metrics <- Filter(function(metric) {
                      !grepl("^val_", metric)
                    }, self$params$metrics)
                  }

                  # initialize metrics & weights
                  for (metric in self$params$metrics)
                    self$metrics[[metric]] <- numeric()

                  # handle metrics
                  if (length(logs) > 0)
                  {
                    self$on_metrics(logs, 0.5)
                  }

                  if (is_run_active()) {
                    self$write_params(self$params)
                    self$write_model_info(self$model)
                  }
                },

                on_epoch_end = function(epoch, logs = NULL) {

                  # handle metrics
                  if(epoch==0){
                    self$weights_last_layer <-
                      coefkeras(self$model)
                  }else{
                    self$weights_last_layer <-
                      cbind(self$weights_last_layer,
                            coefkeras(self$model))
                  }
                  self$on_metrics(logs, 0.1)

                },

                on_metrics = function(logs, sleep) {

                  # record metrics
                  for (metric in names(self$metrics)) {
                    # guard against metrics not yet available by using NA
                    # when a named metrics isn't passed in 'logs'
                    if(metric=="weight_diff"){
                      nc <- NCOL(self$weights_last_layer)
                      if(nc==1){
                        value <- NA
                      }else{
                        value <- sum(abs(
                          apply(self$weights_last_layer[,nc+(-1:0),drop=FALSE],
                                1, diff)
                        ))
                      }
                      if(is.null(value))
                        value <- NA
                    }else{
                      value <- logs[[metric]]
                      if (is.null(value))
                        value <- NA
                      else
                        value <- mean(value)
                    }
                    self$metrics[[metric]] <- c(self$metrics[[metric]], value)
                  }

                  # create history object and convert to metrics data frame
                  history <- keras:::keras_training_history(self$params,
                                                            self$metrics)
                  metrics <- self$as_metrics_df(history)

                  # view metrics if requested
                  if (self$view_metrics) {

                    # create the metrics_viewer or update if we already have one
                    if (is.null(self$metrics_viewer)) {
                      self$metrics_viewer <- view_run_metrics(metrics)
                    }
                    else {
                      update_run_metrics(self$metrics_viewer, metrics)
                    }

                    # pump events
                    Sys.sleep(sleep)
                  }

                  # record metrics
                  write_run_metadata("metrics", metrics)

                },

                # convert keras history to metrics data frame suitable for plotting
                as_metrics_df = function(history) {

                  # create metrics data frame
                  df <- as.data.frame(history$metrics)

                  # pad to epochs if necessary
                  pad <- history$params$epochs - nrow(df)
                  pad_data <- list()
                  for (metric in history$params$metrics)
                    pad_data[[metric]] <- rep_len(NA, pad)
                  df <- rbind(df, pad_data)

                  # return df
                  df
                },

                write_params = function(params) {
                  properties <- list()
                  properties$samples <- params$samples
                  properties$validation_samples <- params$validation_samples
                  properties$epochs <- params$epochs
                  properties$batch_size <- params$batch_size
                  write_run_metadata("properties", properties)
                },

                write_model_info = function(model) {
                  tryCatch({
                    model_info <- list()
                    model_info$model <- py_str(model, line_length = 80L)
                    if (is.character(model$loss))
                      model_info$loss_function <- model$loss
                    else if (inherits(model$loss, "python.builtin.function"))
                      model_info$loss_function <- model$loss$`__name__`
                    optimizer <- model$optimizer
                    if (!is.null(optimizer)) {
                      model_info$optimizer <- py_str(optimizer)
                      model_info$learning_rate <- k_eval(optimizer$lr)
                    }
                    write_run_metadata("properties", model_info)
                  }, error = function(e) {
                    warning("Unable to log model info: ", e$message, call. = FALSE)
                  })

                }
              )
  )

normalize_callbacks_with_metrics_custom <- function (view_metrics, callbacks)
{
  if (!is.null(callbacks) && !is.list(callbacks))
    callbacks <- list(callbacks)
  callbacks <- append(callbacks, KerasMetricsCallback_custom$new(view_metrics))
  normalize_callbacks(callbacks)
}
