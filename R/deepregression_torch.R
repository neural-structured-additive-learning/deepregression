#' Function to initialize a nn_module
#' Forward functions works with a list. The entries of the list are the input of 
#' the subnetworks
#' 
#' @param submodules_list list; subnetworks of distribution parameter
#' @return nn_module
#' @export

model_torch <-  function(submodules_list){
  nn_module(
    classname = "torch_model",
    initialize = function() {
      self$help_forward <- get_help_forward_torch(submodules_list)
      self$subnetwork <- nn_module_dict(submodules_list)
      
    },
    
    forward = function(dataset_list) {
      subnetworks <- lapply(seq_len(length(self$help_forward)),
                            function(x){
                              used_layer <- self$help_forward[[x]]
                              self$subnetwork[[used_layer]](dataset_list[[x]])
      })
      
      Reduce(f = torch_add, subnetworks)
    }
  )}




#' @title Compile a Deep Distributional Regression Model (Torch)
#'
#'
#' @param list_pred_param list of output(-lists) generated from
#' \code{subnetwork_init}
#' @param weights vector of positive values; optional (default = 1 for all observations)
#' @param optimizer optimizer used. Per default Adam
#' @param model_fun which function to use for model building (default \code{keras_model})
#' @param monitor_metrics Further metrics to monitor
#' @param from_preds_to_output function taking the list_pred_param outputs
#' and transforms it into a single network output
#' @param loss the model's loss function; per default evaluated based on
#' the arguments \code{family} and \code{weights} using \code{from_dist_to_loss}
#' @param additional_penalty a penalty that is added to the negative log-likelihood;
#' must be a function of model$trainable_weights with suitable subsetting
#' @param ... arguments passed to \code{from_preds_to_output}
#' @return a list with input tensors and output tensors that can be passed
#' to, e.g., \code{keras_model}
#'
torch_dr <- function(
    list_pred_param,
    optimizer = torch::optim_adam,
    model_fun = NULL, 
    monitor_metrics = list(),
    from_preds_to_output = from_preds_to_dist_torch,
    loss = from_dist_to_loss_torch(family = list(...)$family,
                                   weights = weights),
    additional_penalty = NULL,
    ...
){
  
  out <- from_preds_to_output(list_pred_param, ...)
  # define model
  model <- out
  
  # compile model
  model <- out %>% luz::setup(optimizer = optimizer,
                                loss = loss,
                                metrics = monitor_metrics)
  return(model)
  
}

#' @title Define Predictor of a Deep Distributional Regression Model
#'
#'
#' @param list_pred_param list of output(-lists) generated from
#' \code{subnetwork_init}
#' @param family see \code{?deepregression}; if NULL, concatenated
#' \code{list_pred_param} entries are returned (after applying mapping if provided)
#' @param output_dim dimension of the output
#' @param mapping a list of integers. The i-th list item defines which element
#' elements of \code{list_pred_param} are used for the i-th parameter.
#' For example, \code{mapping = list(1,2,1:2)} means that \code{list_pred_param[[1]]}
#' is used for the first distribution parameter, \code{list_pred_param[[2]]} for
#' the second distribution parameter and  \code{list_pred_param[[3]]} for both
#' distribution parameters (and then added once to \code{list_pred_param[[1]]} and
#' once to \code{list_pred_param[[2]]})
#' @param from_family_to_distfun function to create a \code{dist_fun} 
#' (see \code{?distfun_to_dist}) from the given character \code{family}
#' @param from_distfun_to_dist function creating a tfp distribution based on the
#' prediction tensors and \code{dist_fun}. See \code{?distfun_to_dist}
#' @param add_layer_shared_pred layer to extend shared layers defined in \code{mapping}
#' @param trafo_list a list of transformation function to convert the scale of the
#' additive predictors to the respective distribution parameter
#' @return a list with input tensors and output tensors that can be passed
#' to, e.g., \code{keras_model}
#'
#' @export
from_preds_to_dist_torch <- function(
    list_pred_param, family = NULL,
    output_dim = 1L,
    mapping = NULL, # not implemented
    from_family_to_distfun = make_torch_dist,
    from_distfun_to_dist = distfun_to_dist_torch,
    add_layer_shared_pred = function(input_shape, units) 
      layer_dense_torch(input_shape = input_shape, units = units,
                        use_bias = FALSE),
    trafo_list = NULL, mapping_now = F){
  
  # i think its better to do combination later,
  # because if i do this way i have to load data for each parameter 
  if(mapping_now){
    if(!is.null(mapping)){
      
      lpp <- list_pred_param
      list_pred_param <- list()
      nr_params <- max(unlist(mapping))
      
      if(!is.null(add_layer_shared_pred)){
        
        len_map <- sapply(mapping, length)
        multiple_param <- which(len_map>1)
        
        for(ind in multiple_param){
          amount_layers <- length(lpp[[ind]][[1]]$children)
          # add units (not implemented JUST COPIED)
          if(lpp[[ind]][[1]][[amount_layers]]$weight$shape[1] < len_map[ind]){
            # less units than needed => add layer and then split
            
            #input_shape <- lpp[[ind]][[1]][[amount_layers]]$weight$size()[2]
            #lpp[[ind]] <- torch_cat(
            #  tensors = list(lpp[[ind]][[1]][[amount_layers]]$weight,
            #                add_layer_shared_pred(input_shape = input_shape,
            #                                       units = len_map[ind]*output_dim)
            #                 ))
          } else if(lpp[[ind]][[1]][[amount_layers]]$weight$shape[1] == 
                    len_map[ind]){
            # not sure if good idea
            # maybe better add in model_torch mapping
            lpp[[ind]] <- lapply(seq_len(len_map[ind]), function(x){
              split_torch(module = lpp[[ind]][[1]], index = x)})
            
          }else{
            # more units than needed
            stop("Node ", lpp[[ind]]$name, " has more units than defined by the mapping.\n",
                 "  Does your deep neural network has the correct output dimensions?")
          }
          
        }
        
        lpp <- unlist(lpp, recursive = FALSE)
        mapping <- as.list(unlist(mapping))
        
      }
      
      # store names as they are obstructive later
      names_lpp <- names(lpp)
      lpp <- unname(lpp)
      
      for(i in 1:nr_params){
        list_pred_param[[i]] <- unlist(
          lpp[which(sapply(mapping, function(mp) i %in% mp))])
      }
      
      if(!is.null(names_lpp)) names(list_pred_param) <- names_lpp[1:nr_params]
      
    }}else{
      
      nr_params <- length(list_pred_param)
      
    }
  
  # check family
  if(!is.null(family)){
    if(is.character(family)){
      dist_fun <- from_family_to_distfun(family, output_dim = output_dim,
                                         trafo_list = trafo_list)
    }
  } else{ # assuming that family is a dist_fun already
    dist_fun <- family
  } 
  
  nrparams_dist <- attr(dist_fun, "nrparams_dist")
  if(is.null(nrparams_dist)) nrparams_dist <- nr_params
  
  if(nrparams_dist < nr_params & is.null(mapping))
  {
    warning("More formulas specified than parameters available.",
            " Will only use ", nrparams_dist, " formula(e).")
    nr_params <- nrparams_dist
    list_pred_param <- list_pred_param[1:nrparams_dist]
  }else if(nrparams_dist > nr_params){
    stop("Length of list_of_formula (", nr_params,
         ") does not match number of distribution parameters (",
         nrparams_dist, ").")
  }
  
  if(is.null(names(list_pred_param))){
    names(list_pred_param) <- names_families(family)
  }

  # generate output
  out <- from_distfun_to_dist_torch(dist_fun, list_pred_param)
  
}

#' @title Function to define output distribution based on dist_fun
#'
#' @param dist_fun a distribution function as defined by \code{make_torch_dist}
#' @param preds tensors with predictions
#' @return a symbolic torch distribution
#' @export
#'
from_distfun_to_dist_torch <- function(dist_fun, preds){
  nn_module(
    
    initialize = function() {
      
      self$distr_parameters <- nn_module_dict(
        lapply(preds, function(x) x()))
      self$amount_distr_parameters <- length(preds)
      #names(self$distr_parameters$.__enclos_env__$private$modules_) <- 
      #  names(preds)
    },
    
    forward = function(dataset_list) {
      distribution_parameters <- lapply(seq_len(self$amount_distr_parameters),
                                        function(x){
                                          self[[1]][[x]](dataset_list[[x]])
        })
      
      if(any(names(self[[1]]$.__enclos_env__$private$modules_) == "both")){
        distribution_parameters <- torch_sum(
          torch_stack(
            list(
              torch_cat(distribution_parameters[-length(dataset_list)], dim = 2),
              distribution_parameters[[length(dataset_list)]])
          ), dim = 1)
        
        distribution_parameters <- 
          torch_split(distribution_parameters, split_size = 1, dim = 2) }
      
      dist_fun(distribution_parameters)
    }
  )
}

#' Function to transform a distribution layer output into a loss function
#'
#' @param family see \code{?deepregression}
#' @param weights sample weights
#'
#' @return loss function
from_dist_to_loss_torch <- function(family, weights = NULL){
  
  # define weights to be equal to 1 if not given
  #if(is.null(weights)) weights <- 1
  
  # the negative log-likelihood is given by the negative weighted
  # log probability of the dist
  negloglik <- function(input, target) torch_mean(-input$log_prob(target))
  negloglik
}

