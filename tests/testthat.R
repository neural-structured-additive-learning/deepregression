library(testthat)
library(deepregression)

if (reticulate::py_module_available("tensorflow") & 
    reticulate::py_module_available("keras") & 
    .Platform$OS.type != "windows"){
  test_check("deepregression")
}
