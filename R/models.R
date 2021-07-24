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

