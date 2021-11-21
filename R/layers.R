tib_layer = function(units, la, ...) {
  python_path <- system.file("python", package = "deepregression")
  layers <- reticulate::import_from_path("layers", path = python_path)
  layers$TibLinearLasso(units = units, la = la, ...)
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

tibgroup_layer = function(units, group_idx, la, ...) {
  python_path <- system.file("python", package = "deepregression")
  layers <- reticulate::import_from_path("layers", path = python_path)
  layers$TibGroupLasso(units = units, group_idx = group_idx, la = la, ...)
}