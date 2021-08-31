context("Subnetwork Initilization")

test_that("subnetwork_init", {
  
  form = ~ 1 + d(x) + s(x) + lasso(z) + ridge(z) + d(w) %OZ% (y + s(x)) + d(z) %OZ% s(x) + u
  data = data.frame(x = rnorm(100), y = rnorm(100), 
                    z = rnorm(100), u = rnorm(100),
                    w = rnorm(100))
  controls = penalty_control()
  output_dim = 1L
  param_nr = 1L
  d = dnn_placeholder_processor(function(x) x %>% 
                                  layer_dense(units=5L) %>% 
                                  layer_dense(units=1L))
  specials = c("s", "te", "ti", "vc", "lasso", "ridge", "offset", "vi", "fm", "vfm")
  specials_to_oz = c("d")
  
  
  pp <- processor(form = form, 
                  d = d,
                  specials_to_oz = specials_to_oz, 
                  data = data,
                  output_dim = output_dim,
                  automatic_oz_check = TRUE,
                  param_nr = 1,
                  controls = controls)
  
  res <- subnetwork_init(pp)
  expect_true(all(sapply(res[[1]], function(x) "python.builtin.object" %in% class(x))))
  expect_true("python.builtin.object" %in% class(res[[2]]))
  
})

test_that("helpers subnetwork_init", {
  
  a <- tf$keras$Input(list(3L))
  b <- tf$keras$Input(list(3L))
  c <- tf$keras$Input(list(1L))
  d <- tf$keras$Input(list(1L))
  e <- tf$keras$Input(list(1L))
  
  ktclass <- "keras.engine.keras_tensor.KerasTensor"
  expect_dim <- function(kt, dim){
    expect_equal(kt$shape[[2]], dim)
  }
  
  # layer_add_identity
  expect_error(layer_add_identity(a))
  expect_is(layer_add_identity(list(a)), ktclass)
  expect_is(layer_add_identity(list(c,d,e)), ktclass)
  expect_dim(layer_add_identity(list(a)), 3)
  expect_dim(layer_add_identity(list(a,b)), 3)
  expect_dim(layer_add_identity(list(c,d,e)), 1)
  
  # layer_concatenate_identity
  expect_error(layer_concatenate_identity(a))
  expect_is(layer_concatenate_identity(list(a)), ktclass)
  expect_is(layer_concatenate_identity(list(c,d,e)), ktclass)
  expect_dim(layer_concatenate_identity(list(a)), 3)
  expect_dim(layer_concatenate_identity(list(a,b,c)), 7)
  
})