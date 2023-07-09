context("Data Handler Torch")

test_that("loop_through_pfc_and_call_trafo", {
  
  # only structured
  form = ~ 1 + d(x) + s(x) + lasso(z) + ridge(z) + te(y, df = 5) + d(z)  + u
  data = data.frame(x = rnorm(100), y = rnorm(100), z = rnorm(100), u = rnorm(100))
  controls = penalty_control()
  controls$with_layer <- TRUE
  output_dim = 1L
  param_nr = 1L
  d = dnn_placeholder_processor(function(x) nn_linear(in_features = ncol(x), out_features = 1L))
  specials = c("s", "te", "ti", "vc", "lasso", "ridge", "offset", "vi", "fm", "vfm")
  specials_to_oz = c("d")
  gam_terms <- precalc_gam(list(form), data, controls)
  controls$gamdata <- gam_terms
  
  res1 <- suppressWarnings(
    process_terms(form = form, 
                  d = d,
                  specials_to_oz = specials_to_oz, 
                  data = data,
                  output_dim = output_dim,
                  automatic_oz_check = TRUE,
                  param_nr = 1,
                  controls = controls,
                  parsing_options = form_control(), engine = "torch")
  )
  
  ll <- loop_through_pfc_and_call_trafo(list(res1), engine = "torch")
  expect_true(all(!is.null(sapply(ll, dim))))
  expect_is(ll, "list")
  ll <- loop_through_pfc_and_call_trafo(list(res1), data, engine = "torch")
  expect_true(all(!is.null(sapply(ll, dim))))
  expect_is(ll, "list")
  
  # semi-structured
  data <- as.list(data)
  mnist <- dataset_mnist()
  data$arr <- mnist$train$x[1:100,,]
  form = ~ 1 + d(arr) + s(x) + lasso(z) + ridge(z) + te(y, df = 5)+ u
  gam_terms <- precalc_gam(list(form), data, controls)
  controls$gamdata <- gam_terms
  
  res1 <- suppressWarnings(
    process_terms(form = form, 
                  d = d,
                  specials_to_oz = specials_to_oz, 
                  data = data,
                  output_dim = output_dim,
                  automatic_oz_check = TRUE,
                  param_nr = 1, engine = "torch",
                  controls = controls,
                  parsing_options = form_control())
  )
  ll <- loop_through_pfc_and_call_trafo(list(res1), engine = "torch")
  expect_true(all(!is.null(sapply(ll, dim))))
  expect_is(ll, "list")
  ll <- loop_through_pfc_and_call_trafo(list(res1), data, engine = "torch")
  expect_true(all(!is.null(sapply(ll, dim))))
  expect_is(ll, "list")
  
})

test_that("properties of dataset torch", {
  n <- 100
  loc_x <- matrix(runif(n), ncol = 1)
  scale_intercept <- loc_intercept <- matrix(rep(1, n), ncol = 1)
  target <- matrix(runif(n = n), ncol = 1)
  
  data <- c(
    list(list(loc_intercept, loc_x)),
    list(list(scale_intercept)))
  
  mod <- deepregression(y = target, list_of_formulas = 
                   list(loc = ~ 1 + x,
                        scale = ~ 1), data = data.frame("x" = loc_x),
                 engine = "torch")
  
  luz_dataset <- get_luz_dataset(df_list = data, object = mod)
  
  expect_true("deepregression_luz_dataset" %in% class(luz_dataset))
  
  # two parameters
  expect_equal(length(luz_dataset$.getbatch(1)[[1]]), length(data)[[1]])
  
  expect_true(
    luz_dataset$.length() == n
  )
  
  luz_dataset <- get_luz_dataset(df_list = data, target = target, object = mod)
  # two parameters
  expect_equal(length(luz_dataset$.getbatch(1)[[1]]), attr(
    make_torch_dist(mod$init_params$family), "nrparams_dist"))
  
  expect_true(
    luz_dataset$.length() == n
  )
  expect_true(
    length(luz_dataset$target) == n
  )
  expect_true(
    ncol(luz_dataset$target) == 1
  )
  

})









