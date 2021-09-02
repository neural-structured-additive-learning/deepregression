.onLoad <- function(libname, pkgname) { # nocov start
  reticulate::configure_environment(pkgname)
  # check for tensorflow
  # if (!reticulate::py_module_available("tensorflow")) {
  #   keras::install_keras(version = "2.5.0rc0", tensorflow = "2.5.0rc0", 
  #                        extra_packages = c("tfprobability==0.12", "six")) 
  # }
  # catch tf error
  suppressMessages(try(keras::use_implementation("tensorflow"), silent = TRUE))
  # catch TFP error
  suppressMessages(try(invisible(tfprobability::tfd_normal(0,1)), silent = TRUE))
} # nocov end
