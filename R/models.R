default_train_step = function() {
  python_path <- system.file("python", package = "deepregression")
  models <- reticulate::import_from_path("models", path = python_path)
  
  return(models$default_train_step)
}

build_customKeras = function() {
  python_path <- system.file("python", package = "deepregression")
  models <- reticulate::import_from_path("models", path = python_path)
  
  return(models$build_customKeras())
}

tib_layer = function(units, la, ...) {
  python_path <- system.file("python", package = "deepregression")
  layers <- reticulate::import_from_path("layers", path = python_path)
  layers$TibLinearLasso(num_outputs = units, la = la, ...)
}

simplyconnected_layer = function(la, ...) {
  python_path <- system.file("python", package = "deepregression")
  layers <- reticulate::import_from_path("layers", path = python_path)
  layers$SimplyConnected(la = la, ...)
}

inverse_group_lasso_pen = function(la) {
  python_path <- system.file("python", package = "deepregression")
  layers <- reticulate::import_from_path("layers", path = python_path)
  layers$inverse_group_lasso_pen(la = la)
} 

