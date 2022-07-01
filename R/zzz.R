#' @importFrom stats na.omit

.onLoad <- function(libname, pkgname) { # nocov start
  if(!reticulate::py_available())
  {
    res <- suppressMessages(reticulate::configure_environment(pkgname))
    if(res){
      suppressMessages(try(tf$get_logger()$setLevel('ERROR')))
      suppressMessages(try(tf$autograph$set_verbosity(level=0L)))
      suppressMessages(try(keras::use_implementation("tensorflow"), silent = TRUE))
      # catch TFP error
      suppressMessages(try(invisible(tfprobability::tfd_normal(0,1)), silent = TRUE))
    }else{
      tf <<- reticulate::import("tensorflow", delay_load = TRUE)
      keras <<- reticulate::import("keras", delay_load = TRUE)
      tfp <<- reticulate::import("tensorflow_probability", delay_load = TRUE)
    }
    
  } # nocov end
  # options
  options(orthogonalize = TRUE,
          identify_intercept = FALSE,
          cutoff_names = 60
  )
  # catch TFP start-up error
  suppressMessages(try(invisible(tfp$distributions$Normal(0,1)), silent = TRUE))
}
#' Function to check python environment and install necessary packages
#'
#' If you encounter problems with installing the required python modules
#' please make sure, that a correct python version is configured using
#' `py_discover_config` and change the python version if required.
#' Internally uses keras::install_keras.
#'
#' @param force if TRUE, forces the installations
#' @return Function that checks if a Python environment is available
#' and contains TensorFlow. If not the recommended version is installed.
#' @rdname check_and_install
#'
#' @export
check_and_install <- function(force = FALSE) {
  if (!reticulate::py_module_available("tensorflow") || force) {
    keras::install_keras(version = "2.8.0", tensorflow = "2.8.0", 
                         extra_packages = c("tensorflow_probability==0.16", "six")) # nocov
  } else {
    message("Tensorflow found, skipping tensorflow installation!")
    if (!reticulate::py_module_available("tensorflow_probability") || 
        !reticulate::py_module_available("six")) {
      message("Installing pytho modules 'tfprobability' and 'six'") # nocov
      reticulate::py_install(packages = c("tensorflow-probability==0.16", "six")) # nocov
    }
  }
}
