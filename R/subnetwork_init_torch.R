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
subnetwork_init_torch <- function(pp, deep_top = NULL, 
                                  orthog_fun = NULL, 
                                  split_fun = split_model,
                                  shared_layers = NULL,
                                  param_nr = 1,
                                  selectfun_in = function(pp) pp[[param_nr]],
                                  selectfun_lay = function(pp) pp[[param_nr]],
                                  gaminputs,
                                  summary_layer = layer_add_identity)
{
  
  # subnetwork builder for torch is still rudimentary. It only initializes the
  # different layers and names them
  # Main difference to the tensorflow approach is that the builder don't has a 
  # input output flow. So the best idea is to maintain different subnetwork 
  # builder for the approaches.
  
  pp_in <- selectfun_in(pp)
  pp_lay <- selectfun_lay(pp)
  
  
  layer_matching <- 1:length(pp_in)
  names(layer_matching) <- layer_matching
  
  if(!is.null(shared_layers))
  {
    
    names_terms <- get_names_pfc(pp_in)
    
    for(group in shared_layers){
      
      layer_ref_nr <- which(names_terms==group[1])
      layer_opts <- get("layer_args", environment(pp_lay[[layer_ref_nr]]$layer))
      layer_opts$name <- paste0("shared_",paste(make_valid_layername(group), collapse = ""))
      layer_ref <- do.call(get("layer_class", environment(pp_lay[[layer_ref_nr]]$layer)),
                           layer_opts)
      
      terms_replace_layer <- which(names_terms%in%group)
      layer_matching[terms_replace_layer] <- layer_ref_nr
      
      for(i in terms_replace_layer) {
        pp_lay[[i]]$layer <- function() layer_ref
        pp_lay[[i]]$term <- layer_opts$name
      }
    }
  }
  
  
  if(all(sapply(pp_in, function(x) is.null(x$right_from_oz)))){ 
    # if there is no term to orthogonalize
    outputs <- lapply(1:length(pp_in), function(i){ 
      pp_lay[[i]]$layer()})
    
    names(outputs) <- paste(sapply(1:length(pp_in),
                             function(i) pp_lay[[i]]$term), param_nr, sep = "_")
    outputs
    model_torch(outputs)
  } else{
    
    stop("Orthogonalization not implemented for torch")
  }
}