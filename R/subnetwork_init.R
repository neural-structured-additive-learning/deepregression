#' Initializes a Subnetwork based on the Processed Additive Predictor
#' 
#' @param pp processed predictor list from \code{processor}
#' @param deep_top keras layer if the top part of the deep network after orthogonalization
#' is different to the one extracted from the provided network 
#' @param orthog_fun function used for orthogonalization
#' @param split_fun function to split the network to extract head
#' @param param_nr integer number for the distribution parameter
#' @return returns a list of input and output for this additive predictor
#' 
subnetwork_init <- function(pp, deep_top = NULL, 
                            orthog_fun = orthog_tf, 
                            split_fun = split_model,
                            param_nr = 1)
{
  
  
  inputs <- lapply(pp, function(ap) tf$keras$Input(
    shape = list(as.integer(ap$input_dim)),
    name = paste0("input_", make_valid_layername(ap$term),
                  "_", param_nr))
  )
  
  if(all(sapply(pp, function(x) is.null(x$right_from_oz)))){ # if there is no term to orthogonalize
    
    outputs <- lapply(1:length(pp), function(i) pp[[i]]$layer(inputs[[i]]))
    outputs <- layer_add_identity(outputs)
    return(list(inputs, outputs))
  
  }else{
    
    # define the different types of elements
    outputs_w_oz <- unique(unlist(sapply(pp, "[[", "right_from_oz")))
    outputs_used_for_oz <- which(sapply(pp, function(x) !is.null(x$right_from_oz)))
    outputs_onlyfor_oz <- outputs_used_for_oz[!sapply(pp[outputs_used_for_oz], "[[", "left_from_oz")]
    outputs_wo_oz <- setdiff(1:length(pp), c(outputs_w_oz, outputs_onlyfor_oz))
    
    outputs <- list()
    if(length(outputs_wo_oz)>0) outputs <- layer_add_identity(lapply((1:length(pp))[outputs_wo_oz], 
                                                                     function(i) pp[[i]]$layer(inputs[[i]])))
    ox_outputs <- list()
    k <- 1
    
    for(i in outputs_w_oz){
      
      inputs_for_oz <- which(sapply(pp, function(ap) i %in% ap$right_from_oz))
      ox <- layer_concatenate_identity(inputs[inputs_for_oz])
      if(is.null(deep_top)){
        deep_splitted <- split_fun(pp[[i]]$layer)
      }else{
        deep_splitted <- list(pp[[i]]$layer, deep_top)
      }
    
      deep <- deep_splitted[[1]](inputs[[i]])
      ox_outputs[[k]] <- deep_splitted[[2]](orthog_fun(deep, ox))
      k <- k + 1

    }
    
    if(length(ox_outputs)>0) outputs <- layer_add_identity(c(outputs, ox_outputs))
     
    return(list(inputs, outputs))
     
  }
  
  
}

layer_add_identity <- function(inputs)
{
  
  if(length(inputs)==1) return(inputs[[1]])
  return(tf$keras$layers$add(inputs))
  
}

layer_concatenate_identity <- function(inputs)
{
  
  if(length(inputs)==1) return(inputs[[1]])
  return(tf$keras$layers$concatenate(inputs))
  
}
