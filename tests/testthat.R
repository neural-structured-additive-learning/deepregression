library(testthat)
library(deepregression)

if (reticulate::py_module_available("tensorflow") & 
    reticulate::py_module_available("keras")){
  test_check("deepregression")
}