#' NODE/ODTs Layer
#' 
#' @param name name of the layer 
#' @param units number of output dimensions, for regression and binary 
#' classification: 1, for mc-classification simply the number of classes
#' @param n_layers number of layers consisting of ODTs in NODE
#' @param n_trees number of trees per layer
#' @param tree_depth depth of tree per layer
#' @param threshold_init_beta ??
#' @return layer/model object
#' @export
#' @examples
#' n <- 1000
#' data_regr <- data.frame(matrix(rnorm(4 * n), c(n, 4)))
#' colnames(data_regr) <- c("x0", "x1", "x2", "x3")
#' y_regr <- rnorm(n) + data_regr$x0^2 + data_regr$x1 + 
#'   data_regr$x2*data_regr$x3 + data_regr$x2 + data_regr$x3
#' 
#' library(deepregression)
#' 
#' formula_node <- ~ node(x1, x2, x3, x0, n_trees = 2, n_layers = 2, tree_depth = 2)
#' 
#' mod_node_regr <- deepregression(
#' list_of_formulas = list(loc = formula_node, scale = ~ 1),
#' data = data_regr,
#' y = y_regr
#' )
#' 
#' mod_node_regr %>% fit(epochs = 15, batch_size = 64, verbose = TRUE, 
#'   validation_split = 0.1, early_stopping = TRUE)
#' mod_node_regr %>% predict()
#' 
layer_node <- function(name,
                       units,
                       n_layers = 1L,
                       n_trees = 1L,
                       tree_depth = 1L,
                       threshold_init_beta = 1) {
  python_path <- system.file("python", package = "deepregression")
  node <- reticulate::import_from_path("node", path = python_path)
  return(
    node$layer_node(
      units = units,
      n_layers = n_layers,
      n_trees = n_trees,
      tree_depth = tree_depth,
      threshold_init_beta = threshold_init_beta
    )
  )
}

#' Extract node part from wrapped term
#'
#' @param term character; node model term
#' @return reduced node model term
#' @export
get_node_term <- function(term)
{
  reduced_term <-  sub("^(.*?),[^,]*=.*", "\\1", term)
  if (!grepl(".*\\)$", reduced_term)) {
    reduced_term <- paste0(reduced_term, ")")
  }
  reduced_term
}

#' Extract property of nodedata
#' @param term term in formula
#' @param what string specifying what to return
#' @return property of the node specification as defined by \code{what}
#' @export
get_nodedata <- function(term, what) {
  if (what == "reduced_term")
    return(get_node_term(term))
  else if (what == "n_layers")
    return(as.integer(
      extractval(
        term,
        "n_layers",
        default_for_missing = T,
        default = 1
      )
    ))
  else if (what == "n_trees")
    return(as.integer(
      extractval(
        term,
        "n_trees",
        default_for_missing = T,
        default = 1
      )
    ))
  else if (what == "tree_depth")
    return(as.integer(
      extractval(
        term,
        "tree_depth",
        default_for_missing = T,
        default = 1
      )
    ))
  else if (what == "threshold_init_beta")
    return(as.integer(
      extractval(
        term,
        "threshold_init_beta",
        default_for_missing = T,
        default = 1
      )
    ))
}