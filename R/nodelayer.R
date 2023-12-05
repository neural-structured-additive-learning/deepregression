layer_node <- function(name, units,
                       n_layers = 1L,
                       link = tf$identity,
                       n_trees = 1L,
                       tree_depth = 1L,
                       threshold_init_beta = 1) {
  python_path <- system.file("python", package = "deepregression")
  node <- reticulate::import_from_path("node", path = python_path)
  return(
    node$layer_node(
      units = units,
      n_layers = n_layers,
      link = link, 
      n_trees = n_trees,
      tree_depth = tree_depth,
      threshold_init_beta = threshold_init_beta
    )
  )
}