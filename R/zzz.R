#' @importFrom stats na.omit

VERSIONPY = "3.10"
VERSIONTF = "2.10"
VERSIONKERAS = "2.10"
VERSIONTFP = "0.16"

create_package_name <- function(package, version)
  paste(package, version, sep="==")

.onLoad <- function(libname, pkgname) { # nocov start
  if(suppressMessages(!reticulate::py_available()))
  {
    res <- suppressMessages(reticulate::configure_environment(pkgname))
    if(res & requireNamespace("tensorflow", quietly = TRUE) & 
       requireNamespace("keras", quietly = TRUE)){
      suppressMessages(try(tf$compat$v1$logging$set_verbosity(
        tf$compat$v1$logging$ERROR)))
      suppressMessages(try(tf$get_logger()$setLevel('ERROR')))
      suppressMessages(try(tf$autograph$set_verbosity(level=0L)))
      suppressMessages(try(keras::use_implementation("tensorflow"), silent = TRUE))
      # catch TFP error
      suppressMessages(try(invisible(tfprobability::tfd_normal(0,1)), silent = TRUE))
      suppressMessages(try(invisible(tfprobability::tfd_normal(0,1)), silent = TRUE))
    }else{
      tf <<- reticulate::import("tensorflow", delay_load = TRUE)
      keras <<- reticulate::import("keras", delay_load = TRUE)
      tfp <<- reticulate::import("tensorflow_probability", delay_load = TRUE)
    }
    
  }else{
    tf <- reticulate::import("tensorflow")
  } # nocov end
  # options
  options(orthogonalize = TRUE,
          identify_intercept = FALSE,
          cutoff_names = 60
  )
}
#' Function to check python environment and install necessary packages
#'
#' If you encounter problems with installing the required python modules
#' please make sure, that a correct python version is configured using
#' \code{py_discover_config} and change the python version if required.
#' Internally uses \code{keras::install_keras}.
#'
#' @param force if TRUE, forces the installations
#' @return Function that checks if a Python environment is available
#' and contains TensorFlow. If not the recommended version is installed.
#' @rdname check_and_install
#'
#' @export
check_and_install <- function(force = FALSE) {
  if (!reticulate::py_module_available("tensorflow") || force) {
    keras::install_keras(version = VERSIONKERAS, tensorflow = VERSIONTF, 
                         extra_packages = c(create_package_name("tensorflow_probability", VERSIONTFP),
                                            "six")) # nocov
  } else {
    message("Tensorflow found, skipping tensorflow installation!")
    if (!reticulate::py_module_available("tensorflow_probability") || 
        !reticulate::py_module_available("six")) {
      message("Installing python modules 'tfprobability' and 'six'") # nocov
      reticulate::py_install(packages = c(create_package_name("tensorflow_probability", VERSIONTFP), 
                                          "six")) # nocov
    }
  }
}

#' Function to update miniconda and packages
#' 
#' @param python string; version of python
#' @param uninstall logical; whether to uninstall previous conda env
#' @param also_packages logical; whether to install also all required packages
#' 
update_miniconda_deepregression <- function(python = VERSIONPY, 
                                            uninstall = TRUE,
                                            also_packages = TRUE)
{
  
  if(uninstall) reticulate::miniconda_uninstall()
  
  Sys.setenv(RETICULATE_MINICONDA_PYTHON_VERSION=python)
  reticulate::install_miniconda()
  
  if(also_packages){

    reticulate::conda_install(packages=c(create_package_name("tensorflow", VERSIONTF),
                             create_package_name("keras", VERSIONKERAS),
                             create_package_name("tensorflow-probability", VERSIONTFP)),
                  python_version = VERSIONPY)
    # tensorflow::install_tensorflow(version = VERSIONTF)
    # keras::install_keras(version = VERSIONKERAS, tensorflow = VERSIONTF,
    #                      extra_packages = c(create_package_name("tensorflow_probability", VERSIONTFP),
    #                                         "six"))
    
  }
  # maybe requires a final sudo apt-get dist-upgrade
  
}