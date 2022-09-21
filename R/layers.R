#' Hadamard-type layers
#' 
#' @param units integer; number of units
#' @param la numeric; regularization value (> 0)
#' @param ... arguments passed to TensorFlow layer
#' @return layer object
#' @export
#' @rdname hadamard_layers
tib_layer = function(units, la, ...) {
  python_path <- system.file("python", package = "deepregression")
  layers <- reticulate::import_from_path("layers", path = python_path)
  layers$TibLinearLasso(units = units, la = la, ...)
}

#' @export
#' @rdname hadamard_layers
simplyconnected_layer = function(la, ...) {
  python_path <- system.file("python", package = "deepregression")
  layers <- reticulate::import_from_path("layers", path = python_path)
  layers$SimplyConnected(la = la, ...)
}

#' @export
#' @rdname hadamard_layers
inverse_group_lasso_pen = function(la) {
  python_path <- system.file("python", package = "deepregression")
  layers <- reticulate::import_from_path("layers", path = python_path)
  layers$inverse_group_lasso_pen(la = la)
} 

#' @param group_idx list of group indices
#' @export
#' @rdname hadamard_layers
regularizer_group_lasso = function(la, group_idx) {
  python_path <- system.file("python", package = "deepregression")
  layers <- reticulate::import_from_path("layers", path = python_path)
  layers$ExplicitGroupLasso(la = la, group_idx = group_idx)
}

#' @export
#' @rdname hadamard_layers
tibgroup_layer = function(units, group_idx, la, ...) {
  python_path <- system.file("python", package = "deepregression")
  layers <- reticulate::import_from_path("layers", path = python_path)
  layers$TibGroupLasso(units = units, group_idx = group_idx, la = la, ...)
}

#' @export
#' @rdname hadamard_layers
layer_hadamard = function(units, la, depth, ...) {
  python_path <- system.file("python", package = "deepregression")
  layers <- reticulate::import_from_path("layers", path = python_path)
  layers$HadamardLayer(units = units, la = la, depth = depth, ...)
}

#' @export
#' @rdname hadamard_layers
layer_group_hadamard = function(units, la, group_idx, depth, ...) {
  python_path <- system.file("python", package = "deepregression")
  layers <- reticulate::import_from_path("layers", path = python_path)
  layers$GroupHadamardLayer(units = units, la = la, group_idx = group_idx, depth = depth, ...)
}

#' @param initu,initv initializers for parameters
#' @export
#' @rdname hadamard_layers
layer_hadamard_diff = function(units, la, initu = "glorot_uniform", initv = "glorot_uniform", ...) {
  python_path <- system.file("python", package = "deepregression")
  layers <- reticulate::import_from_path("layers", path = python_path)
  layers$HadamardDiffLayer(units = units, la = la, initu = initu, initv = initv, ...)
}

#' @param depth integer; depth of weight factorization
#' @rdname hadamard_layers
#' @export
#' 
layer_hadamard = function(units=1, la=0, depth=3, ...) {
  python_path <- system.file("python", package = "deepregression")
  layers <- reticulate::import_from_path("layers", path = python_path)
  layers$HadamardLayer(units = units, la = la, depth = depth, ...)
}

#' Sparse 2D Convolutional layer
#' 
#' @param filters number of filters
#' @param kernel_size size of convolutional filter
#' @param lam regularization strength
#' @param depth depth of weight factorization
#' @param ... arguments passed to TensorFlow layer
#' @return layer object
#' @export
#' 
layer_sparse_conv_2d <- function(filters,
                                 kernel_size,
                                 lam=NULL,
                                 depth=2, ...) {
  python_path <- system.file("python", package = "deepregression")
  layers <- reticulate::import_from_path("layers", path = python_path)
  layers$SparseConv2D(filters = filters, kernel_size = kernel_size, lam = lam, depth = depth, ...)
}
