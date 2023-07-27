#' Helper function to create an function that generates R6 instances of 
#' class dataset
#' @param df_list list; data for the distribution learning model (data for every distributional parameter)
#' @param target vector; target value
#' @param length amount of inputs
#' @param object deepregression object
#' @return R6 instances of class dataset
#' @export
get_luz_dataset <- dataset(
  "deepregression_luz_dataset",
  
  initialize = function(df_list, target = NULL, length = NULL, object) {
    
    
    if(!attr(make_torch_dist(object$init_params$family), "nrparams_dist") == 1 &
       sum(names(object$model()$modules) == "subnetwork") == 1){
      self$getbatch <- self$getbatch_specialcase
      setup_loader <- self$setup_loader_specialcase
    }else{
      self$getbatch <- self$getbatch_normal
      setup_loader <- self$setup_loader_normal
    } 
    
    self$df_list <- df_list
    
    self$data <- setup_loader(df_list)
    self$target <- target
    
    if(!is.null(length)) self$length <- length 
    
    
  },
  
  # has to be fast because is used very often (very sure this a bottle neck)
  .getbatch = function(index) {
    self$getbatch(index)
  },
  
  .length = function() {
    if(!is.null(self$length)) return(self$length)
    return(nrow(self$df_list[[1]][[1]]))
  },
  
  setup_loader_normal = function(df_list){
    
    lapply(df_list, function(x) 
      lapply(x, function(y){
        if((ncol(y)==1) & check_data_for_image(y)){
          return( 
            function(index) torch_stack(
              lapply(index, function(x) y[x, ,drop = F] %>% base_loader() %>%
                       transform_to_tensor())))
          # this torch_stack(...) allows to use .getbatch also when
          # we use image data.
        }
        function(index) torch_tensor(y[index, ,drop = F])
      }))
  },
  
  setup_loader_specialcase = function(df_list){
    df_list <- unlist(df_list, recursive = F)
    
    lapply(df_list, function(y){
        if((ncol(y)==1) & check_data_for_image(y)){
          return( 
            function(index) torch_stack(
              lapply(index, function(x) y[x, ,drop = F] %>% base_loader() %>%
                       transform_to_tensor())))
          # this torch_stack(...) allows to use .getbatch also when
          # we use image data.
        }
        function(index) torch_tensor(y[index, ,drop = F])
      })
  },
  
  getbatch_normal = function(index){
    indexes <- lapply(self$data,
                      function(x) lapply(x, function(y) y(index)))
    if(is.null(self$target)) return(list(indexes))
    target <- self$target[index]
    list(indexes, target)
    },
  
  getbatch_specialcase = function(index){
    
    indexes <- lapply(self$data, function(y) y(index))
    
    if(is.null(self$target)) return(list(indexes))
    target <- self$target[index]
    list(indexes, target)}
)


# till now only orthog_options will be test
# Motivation is to test inputs given engine
check_input_torch <- function(orthog_options){
  if(orthog_options$orthogonalize != FALSE) 
    stop("Orthogonalization not implemented for torch")
}


get_weights_torch <- function(model){
  old_weights <- lapply(model$model()$parameters, function(x) as_array(x))
  lapply(old_weights, function(x) torch_tensor(x))
}


weight_reset <-  function(m) {
  try(m$reset_parameters(), silent = T)
}




# check if variable contains image data

check_data_for_image <- function(data){
  jpg_yes <- all(grepl(x = data, pattern = ".jpg"))
  png_yes <- all(grepl(x = data, pattern = ".png"))
  image_yes <- jpg_yes | png_yes
  image_yes
}


#' Helper function to calculate amount of layers
#' Needed when shared layers are used, because of layers have same names

#' @param list_pred_param list; subnetworks 
#' @return layers
#' @export
get_help_forward_torch <- function(list_pred_param){
  
  layer_names <- names(list_pred_param)
  amount_layer <- length(list_pred_param)
  amount_unique_layers <- seq_len(length(unique(unlist(layer_names))))
  names(amount_unique_layers) <- unique(unlist(layer_names))
  used_layers <- unlist(lapply(layer_names, FUN = function(x) amount_unique_layers[x]))
  
  used_layers
}



#' Function to check if inputs are supported by corresponding fit function
#' @param args list; list of arguments used in fit process 
#' @param fit_fun used fit function (e.g. fit.keras.engine.training.Model)
#' @return stop message if inputs are not supported
#' @export
check_input_args_fit <- function(args, fit_fun){
    if(!all(names(args) %in% formalArgs(fit_fun))){
      not_supported <- names(
        args[which(!(names(args) %in% formalArgs(fit_fun)))])
      stop(sprintf("Following arguments are not supported by fit: %s",
                   not_supported))
      
      }
}



