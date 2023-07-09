

#' Function to define spline as Torch layer
#' 
#' @param units integer; number of output units
#' @param P matrix; penalty matrix
#' @param name string; string defining the layer's name
#' @param trainable logical; whether layer is trainable
#' @param kernel_initializer initializer; for basis coefficients
#' @return Torch layer
#' @export
layer_spline_torch <- function(P, units = 1L, name, trainable = TRUE,
                               kernel_initializer = "glorot_uniform", ...){
  
  P <- torch_tensor(P)
  input_shape <- P$size(1)

  dots <- list(...)
  kernel_initializer <- do.call(
    choose_kernel_initializer_torch, list(kernel_initializer,
                                          dots$kernel_initializer_value))
  
  layer_module <- layer_dense_module(kernel_initializer)
  spline_layer <-  layer_module(in_features = input_shape,
                         out_features = units, bias = F)
  
  spline_layer$parameters$weight$register_hook(function(grad){
    grad + torch_matmul((P+P$t()), spline_layer$weight$t())$t()
  })
  
  if(!trainable) spline_layer$parameters$weight$requires_grad_(F)
  
  spline_layer
}
