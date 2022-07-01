#' Initializes a Subnetwork based on the Processed Additive Predictor
#' 
#' @param pp list of processed predictor lists from \code{processor}
#' @param deep_top keras layer if the top part of the deep network after orthogonalization
#' is different to the one extracted from the provided network 
#' @param orthog_fun function used for orthogonalization
#' @param split_fun function to split the network to extract head
#' @param shared_layers list defining shared weights within one predictor;
#' each list item is a vector of characters of terms as given in the parameter formula
#' @param param_nr integer number for the distribution parameter
#' @param selectfun_in,selectfun_lay functions defining which subset of pp to
#' take as inputs and layers for this subnetwork; per default the \code{param_nr}'s entry
#' @param gaminputs input tensors for gam terms
#' @param summary_layer keras layer that combines inputs (typically adding or concatenating)
#' @return returns a list of input and output for this additive predictor
#' 
#' @export
#' 
subnetwork_init <- function(pp, deep_top = NULL, 
                            orthog_fun = orthog_tf, 
                            split_fun = split_model,
                            shared_layers = NULL,
                            param_nr = 1,
                            selectfun_in = function(pp) pp[[param_nr]],
                            selectfun_lay = function(pp) pp[[param_nr]],
                            gaminputs,
                            summary_layer = layer_add_identity)
{
  
  # instead of passing the respective pp,
  # subsetting is done within subnetwork_init
  # to allow other subnetwork_builder to 
  # potentially access all pp entries
  pp_in <- selectfun_in(pp)
  pp_lay <- selectfun_lay(pp)
  
  # generate pp parts
  gaminput_nrs <- sapply(pp_in, "[[", "gamdata_nr")
  has_gaminp <- !sapply(gaminput_nrs,is.null)
  gaminput_comb <- sapply(pp_in[which(has_gaminp)], "[[", "gamdata_combined")
  inputs <- makeInputs(pp_in, param_nr = param_nr)
  org_inputs_for_concat <- list()
  
  if(sum(has_gaminp)){
    
    for(i in 1:sum(has_gaminp)){
      
      # concatenate inputs or replace?
      concat <- gaminput_comb[[i]]
      nr <- which(has_gaminp)[i]
      
      if(!is.null(concat) && concat){
        
        org_inputs_for_concat <- c(
          org_inputs_for_concat,
          inputs[[nr]]
        )
        inputs[[nr]] <- layer_concatenate_identity(
          list(gaminputs[[gaminput_nrs[[nr]]]], inputs[[nr]])
        )
        
      }else{
        
        inputs[[nr]] <- gaminputs[[gaminput_nrs[[nr]]]]
        
      }
      
    }
    
    inputs_to_replace <- which(has_gaminp)[gaminput_comb]
    keep_inputs_in_return <- setdiff(1:length(inputs), (which(has_gaminp)[!gaminput_comb]))
    
  }else{
    
   inputs_to_replace <-  c()
   keep_inputs_in_return <- 1:length(inputs)
    
  }
  
  layer_matching <- 1:length(pp_in)
  names(layer_matching) <- layer_matching
  
  if(!is.null(shared_layers))
  {
    
    names_terms <- get_names_pfc(pp_in)
    
    for(group in shared_layers){
      
      layer_ref_nr <- which(names_terms==group[1])
      layer_opts <- get("layer_args", environment(pp_lay[[layer_ref_nr]]$layer))
      layer_opts$name <- paste0("shared_", 
                                makelayername(paste(group, collapse="_"), 
                                              param_nr))
      layer_ref <- do.call(get("layer_class", environment(pp_lay[[layer_ref_nr]]$layer)),
                           layer_opts)
      
      terms_replace_layer <- which(names_terms%in%group)
      layer_matching[terms_replace_layer] <- layer_ref_nr
      for(i in terms_replace_layer) pp_lay[[i]]$layer <- layer_ref
      
    }
  }
  
  if(all(sapply(pp_in, function(x) is.null(x$right_from_oz)))){ # if there is no term to orthogonalize
    
    outputs <- lapply(1:length(pp_in), function(i) pp_lay[[layer_matching[i]]]$layer(inputs[[i]]))
    outputs <- summary_layer(outputs)
    
    # replace original inputs
    if(length(org_inputs_for_concat)>0)
      inputs[inputs_to_replace] <- org_inputs_for_concat
    return(list(inputs[keep_inputs_in_return], outputs))
  
  }else{
    
    # define the different types of elements
    outputs_w_oz <- unique(unlist(sapply(pp_in, "[[", "right_from_oz")))
    outputs_used_for_oz <- which(sapply(pp_in, function(x) !is.null(x$right_from_oz)))
    outputs_onlyfor_oz <- outputs_used_for_oz[!sapply(pp_in[outputs_used_for_oz], "[[", "left_from_oz")]
    outputs_wo_oz <- setdiff(1:length(pp_in), c(outputs_w_oz, outputs_onlyfor_oz))
    
    outputs <- list()
    if(length(outputs_wo_oz)>0) outputs <- 
      layer_add_identity(lapply((1:length(pp_in))[outputs_wo_oz], 
                                function(i) pp_lay[[layer_matching[i]]]$layer(inputs[[i]])))
    ox_outputs <- list()
    k <- 1
    
    for(i in outputs_w_oz){
      
      inputs_for_oz <- which(sapply(pp_in, function(ap) i %in% ap$right_from_oz))
      ox <- layer_concatenate_identity(inputs[inputs_for_oz])
      if(is.null(deep_top)){
        deep_splitted <- split_fun(pp_lay[[layer_matching[i]]]$layer)
      }else{
        deep_splitted <- list(pp_lay[[layer_matching[i]]]$layer, deep_top)
      }
    
      deep <- deep_splitted[[1]](inputs[[i]])
      ox_outputs[[k]] <- deep_splitted[[2]](orthog_fun(deep, ox))
      k <- k + 1

    }
    
    if(length(ox_outputs)>0) outputs <- summary_layer(c(outputs, ox_outputs))
    
    if(length(org_inputs_for_concat)>0)
      inputs[inputs_to_replace] <- org_inputs_for_concat
    return(list(inputs[keep_inputs_in_return], outputs))
     
  }
  
  
}
#' Convenience layer function
#' 
#' @param inputs list of tensors
#' @return tensor
#' @details convenience layers to work with list of inputs where \code{inputs}
#' can also have length one
#' 
#' @export
#' @rdname convenience_layers
#' 
#' 
layer_add_identity <- function(inputs)
{
  
  if(length(inputs)==1) return(inputs[[1]])
  return(tf$keras$layers$add(inputs))
  
}

#' @export
#' @rdname convenience_layers
layer_concatenate_identity <- function(inputs)
{
  
  if(length(inputs)==1) return(inputs[[1]])
  return(tf$keras$layers$concatenate(inputs))
  
}


#' Convenience layer function
#' 
#' @param pp processed predictors
#' @param param_nr integer for the parameter
#' @return input tensors with appropriate names
#' 
#' @export
#' 
#' 
makeInputs <- function(pp, param_nr)
{
 
  lapply(pp, function(ap){ 
    
    if(length(ap$input_dim)>1)
      inp <- as.list(as.integer(ap$input_dim)) else
        inp <- list(as.integer(ap$input_dim))
      
      return(
        tf$keras$Input(
          shape = inp,
          name = paste0("input_", strtrim(make_valid_layername(ap$term), options()$cutoff_names),
                        "_", param_nr))
      )
  }
  ) 
  
}