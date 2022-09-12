build_customKeras = function(...) {
  python_path <- system.file("python", package = "deepregression")
  models <- reticulate::import_from_path("models", path = python_path)
  
  return(models$build_customKeras(...))
}

#' Function to define an optimizer combining multiple optimizers
#' @param optimizers_and_layers a list if \code{tuple}s of optimizer
#' and respective layers
#' @return an optimizer
#' @export
multioptimizer = function(optimizers_and_layers) {
  python_path <- system.file("python", package = "deepregression")
  opt <- reticulate::import_from_path("optimizers", path = python_path)
  
  return(opt$MultiOptimizer(optimizers_and_layers))
}

