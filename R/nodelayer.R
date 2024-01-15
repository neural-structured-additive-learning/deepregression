layer_node <- function(name,
                       units,
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

#' Extract node part from wrapped term
#'
#' @param term character; node model term
#' @return reduced node model term
#' @export
get_node_term <- function(term)
{
  #print("get_node_term----------")
  reduced_term <-  sub("^(.*?),[^,]*=.*", "\\1", term)
  #print(reduced_term)
  if (!grepl(".*\\)$", reduced_term)) {
    reduced_term <- paste0(reduced_term, ")")
  }
  #print(reduced_term)
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
  else if (what == "link")
    return(extractval(
      term,
      "link",
      default_for_missing = T,
      default = tf$identity
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