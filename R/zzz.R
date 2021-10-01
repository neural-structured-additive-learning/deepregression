.onLoad <- function(libname, pkgname) { # nocov start
  reticulate::configure_environment(pkgname)
  suppressMessages(try(keras::use_implementation("tensorflow"), silent = TRUE))
  # catch TFP error
  suppressMessages(try(invisible(tfprobability::tfd_normal(0,1)), silent = TRUE))
  # options
  options(orthogonalize = TRUE,
          identify_intercept = FALSE
  )
} # nocov end

