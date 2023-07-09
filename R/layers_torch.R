
layer_dense_module <- function(kernel_initializer){
  nn_module(classname = "nn_linear_deepregression",
            initialize = nn_linear$public_methods$initialize,
            forward = nn_linear$public_methods$forward,
            reset_parameters = kernel_initializer
            )
}

# nn_init constand does not work if value is more than 1
# so warmstarts for gam doesnot wokr
nn_init_no_grad_constant_deepreg <- function(tensor, value){
  
  if(length(value) == 1){
    with_no_grad({
      tensor$fill_(value)
    })
    return(tensor)
  }
  
  with_no_grad({
    tensor <- tensor$t()
    lapply(1:length(value),
           function(x) tensor[x] = tensor$index_fill(1, x, value[x])[x])
    tensor <- tensor$t()
  })
  tensor
}


#' Function to define a torch layer similar to a tf dense layer
#' 
#' @param units integer; number of output units
#' @param name string; string defining the layer's name
#' @param trainable logical; whether layer is trainable
#' @param kernel_initializer initializer; for coefficients
#' @return Torch layer
#' @export
layer_dense_torch <- function(input_shape, units = 1L, name, trainable = TRUE,
                              kernel_initializer = "glorot_uniform",
                              use_bias = FALSE, kernel_regularizer = NULL, ...){
  
  dots <- list(...)
  kernel_initializer <- do.call(
    choose_kernel_initializer_torch, list(kernel_initializer,
                                          dots$kernel_initializer_value))
  
  layer_module <- layer_dense_module(kernel_initializer)
  layer <-  layer_module(in_features = input_shape,
                     out_features = units, bias = use_bias)
  
  if(!trainable) layer$parameters$weight$requires_grad_(F)
  
  
  if(!is.null(kernel_regularizer)){
    if(kernel_regularizer$regularizer == "l2") {
      layer$parameters$weight$register_hook(function(grad){
        grad + 2*(kernel_regularizer$la)*(layer$parameters$weight)
      })
    }}
  
  layer
}

simplyconnected_layer_torch <-
  nn_module(
    classname = "simply_con",
    initialize = function(la = la,  multfac_initializer = torch_ones, 
                          input_shape){
      self$la <- torch_tensor(la)
      self$multfac_initializer <- multfac_initializer
      
      sc <- nn_parameter(x = self$multfac_initializer(input_shape))
      
      sc$register_hook(function(grad) {grad + 2*la*sc})
      self$sc <- sc
      
    },
    
    forward = function(dataset_list){
      torch_multiply(
        self$sc$view(c(1, length(self$sc))),
        dataset_list)
    }
  )


tiblinlasso_layer_torch <- function(la, input_shape = 1, units = 1,
                                    kernel_initializer = "he_normal"){
  
  la <- torch_tensor(la)
  
  kernel_initializer <- choose_kernel_initializer_torch(kernel_initializer)

  tiblinlasso_module <- layer_dense_module(kernel_initializer)
  tiblinlasso_layer <-  tiblinlasso_module(in_features = input_shape,
                         out_features = units, bias = F)
  
  tiblinlasso_layer$parameters$weight$register_hook(function(grad){
    grad + 2*la*tiblinlasso_layer$parameters$weight
  })
  tiblinlasso_layer
}

tib_layer_torch <-
  nn_module(
    classname = "TibLinearLasso_torch",
    initialize = function(units, la, input_shape, multfac_initializer = torch_ones){
      self$units = units
      self$la = la
      self$multfac_initializer = multfac_initializer
      
      self$fc <- tiblinlasso_layer_torch(la = self$la,
                                         input_shape = input_shape,
                                         units = self$units)
      self$sc <- simplyconnected_layer_torch(la = self$la,
                                             input_shape = input_shape,
                                             multfac_initializer = 
                                               self$multfac_initializer)
      
    },
    forward = function(data_list){
      self$fc(self$sc(data_list))
    }
  )

tibgroup_layer_torch <-
  nn_module(
    classname = "TibGroupLasso_torch",
    initialize = function(units, group_idx=NULL, la = 0, input_shape,
                          kernel_initializer = "torch_ones", 
                          multfac_initializer = "he_normal"){
      
      self$units = units
      self$la = la
      self$group_idx <- group_idx
      self$kernel_regularizer$regularizer <- "l2"
      self$kernel_regularizer$la <- la
      self$kernel_initializer = kernel_initializer
      self$multfac_initializer = multfac_initializer
      
      if(is.null(self$group_idx)){
        self$fc <- layer_dense_torch(input_shape = 1, units = 1,
                                     kernel_regularizer = self$kernel_regularizer,
                                     kernel_initializer = self$kernel_initializer,
                                     use_bias = F)
        
        self$gc <- layer_dense_torch(input_shape = input_shape, units = 1,
                                     kernel_regularizer = self$kernel_regularizer,
                                     kernel_initializer = self$multfac_initializer,
                                     use_bias = F)
      } #else not sure
      
      
    },
    forward = function(data_list){
      self$fc(self$gc(data_list))
    }
  )

