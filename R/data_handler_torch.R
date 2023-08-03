#' Function to additionally prepare data for fit process (torch)
#' 
#' @param pfc list of processor transformed formulas 
#' @param input_x output of prepare_data()
#' @param target target values
#' @param object a deepregression object
#' @return list of matrices or arrays for predict or a dataloader for fit process
#' 
#' @importFrom torch torch_tensor

prepare_data_torch <- function(pfc, input_x, target = NULL, object){
  
  distr_datasets_length <- sapply(
    1:length(pfc),
    function(x) length(pfc[[x]])
  )
  
  sequence_length <- seq_len(sum(distr_datasets_length))
  
  distr_datasets_index <- lapply(distr_datasets_length, function(x){
    index <- sequence_length[seq_len(x)]
    sequence_length <<- sequence_length[-seq_len(x)]
    index
  })
  
  df_list <- lapply(distr_datasets_index, function(x){ input_x[x] })
  
  # special case
  special_case <- FALSE
  if(!attr(make_torch_dist(object$init_params$family), "nrparams_dist") == 1){
    if(sum(names(object$model()$modules) == "subnetwork") == 1){ 
      # if is also triggered when binomal or poisson (distribution with one parameter)
      # only be triggered if we have multiple parameters and just want to model one
      helper_df_list <- unlist(lapply( strsplit(names(object$model()$modules), split = "_"), 
                     function(x) tail(x, 1)))
      amount_params <- seq_len(length(object$init_params$parsed_formulas_contents))
      df_list_index <- which(amount_params %in% helper_df_list)
      df_list <- df_list[df_list_index]
      special_case <-  TRUE}
  }
  
  #predict cases
  if(is.null(target) & special_case)  return(unlist(df_list, recursive = F))  
  if(is.null(target)) return(df_list)
  
  get_luz_dataset(df_list = df_list, target = torch::torch_tensor(target)$to(torch::torch_float()),
                  object = object)
  }


#' Function to prepare input list for fit process, due to different approaches
#' 
#' @param input_x output of prepare_data()
#' @param input_y target
#' @param object a deepregression object
#' @param epochs integer, the number of epochs to fit the model
#' @param batch_size integer, the batch size used for mini-batch training
#' @param validation_split float in [0,1] defining the amount of data used for validation
#' @param validation_data optional specified validation data
#' @param callbacks a list of callbacks for fitting
#' @param verbose logical, whether to print losses during training.
#' @param view_metrics logical, whether to trigger the Viewer in RStudio / Browser.
#' @param early_stopping logical, whether early stopping should be user.
#' @return list of arguments used in fit function
#' 
#' @importFrom torch dataloader dataset_subset
#' 
prepare_input_list_model <- function(input_x, input_y,
                                     object, epochs = 10, batch_size = 32,
                                     validation_split = 0, validation_data = NULL,
                                     callbacks = NULL, verbose, view_metrics, 
                                     early_stopping){
  
  if(object$engine == 'tf'){
    
    input_list_model <-
      list(object = object$model,
           epochs = epochs,
           batch_size = batch_size,
           validation_split = validation_split,
           validation_data = validation_data,
           callbacks = callbacks,
           verbose = verbose,
           view_metrics = ifelse(view_metrics,
                                 getOption("keras.view_metrics",
                                           default = "auto"), FALSE)
      )
    
    input_list_model <- c(input_list_model,
                          list(x = input_x, y = input_y))
    
    return(input_list_model)
  } else {
    input_dataloader <- prepare_data_torch(
      pfc  = object$init_params$parsed_formulas_content,
      input_x = input_x,
      target = input_y,
      object = object)
    
    #no validation
    if(is.null(validation_data) & identical(validation_split, 0)){
      train_dl <- torch::dataloader(input_dataloader, batch_size = batch_size)
      valid_dl <- NULL
    }
    
    if(!is.null(validation_data)){
      
      train_dl <- torch::dataloader(input_dataloader, batch_size = batch_size,
                             shuffle = T)
      
      validation_dataloader <- prepare_data_torch(
        pfc  = object$init_params$parsed_formulas_content,
        input_x = validation_data[[1]],
        target = validation_data[[2]],
        object = object)
      
      valid_dl <- torch::dataloader(validation_dataloader, batch_size = batch_size,
                             shuffle = T)
    }
    
    if(!identical(validation_split, 0) & !is.null(validation_split)){
      
      train_ids <- 1:(ceiling((1-validation_split) * length(input_y)))
      valid_ids <- setdiff(1:length(input_y), train_ids)

      train_ds <- torch::dataset_subset(input_dataloader, indices = train_ids)
      valid_ds <- torch::dataset_subset(input_dataloader, indices = valid_ids)
      
      if(any(unlist(lapply(input_x, check_data_for_image)))) 
        cat(sprintf("Found %s validated image filenames \n", length(train_ids)))
      train_dl <- torch::dataloader(train_ds, batch_size = batch_size, shuffle = T)
      
      if(any(unlist(lapply(input_x, check_data_for_image))))
        cat(sprintf("Found %s validated image filenames \n", length(valid_ids)))
      valid_dl <- torch::dataloader(valid_ds, batch_size = batch_size, shuffle = T)
    }
    
    if(!is.null(valid_dl)) valid_data <- valid_dl
    if(is.null(valid_dl)) valid_data <- NULL
    
    input_list_model <- list(
      object = object$model,
      epochs = epochs,
      data = train_dl,
      valid_data = valid_data,
      callbacks = callbacks,
      verbose = verbose)
    
    c(input_list_model)
  }
  
  
}