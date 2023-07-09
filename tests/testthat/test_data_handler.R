context("Data Handler")

test_that("loop_through_pfc_and_call_trafo", {
  
  # only structured
  form = ~ 1 + d(x) + s(x) + lasso(z) + ridge(z) + te(y, df = 5) %OZ% (y + s(x)) + d(z) %OZ% s(x) + u
  data = data.frame(x = rnorm(100), y = rnorm(100), z = rnorm(100), u = rnorm(100))
  controls = penalty_control()
  controls$with_layer <- TRUE
  output_dim = 1L
  param_nr = 1L
  d = dnn_placeholder_processor(function(x) layer_dense(x, units=1L))
  specials = c("s", "te", "ti", "vc", "lasso", "ridge", "offset", "vi", "fm", "vfm")
  specials_to_oz = c("d")
  gam_terms <- precalc_gam(list(form), data, controls)
  controls$gamdata <- gam_terms
  
  res1 <- suppressWarnings(
    process_terms(form = form, 
                  d = dnn_placeholder_processor(function(x) x %>% layer_dense(x, units=1L)),
                  specials_to_oz = specials_to_oz, 
                  data = data,
                  output_dim = output_dim,
                  automatic_oz_check = TRUE,
                  param_nr = 1,
                  controls = controls,
                  parsing_options = form_control())
  )
  
  ll <- loop_through_pfc_and_call_trafo(list(res1))
  expect_true(all(!is.null(sapply(ll, dim))))
  expect_is(ll, "list")
  ll <- loop_through_pfc_and_call_trafo(list(res1), data)
  expect_true(all(!is.null(sapply(ll, dim))))
  expect_is(ll, "list")
  
  # semi-structured
  data <- as.list(data)
  mnist <- dataset_mnist()
  data$arr <- mnist$train$x[1:100,,]
  form = ~ 1 + d(arr) + s(x) + lasso(z) + ridge(z) + te(y, df = 5) %OZ% (y + s(x)) + d(z) %OZ% s(x) + u
  gam_terms <- precalc_gam(list(form), data, controls)
  controls$gamdata <- gam_terms
  
  res1 <- suppressWarnings(
    process_terms(form = form, 
                  d = dnn_placeholder_processor(function(x) x %>% layer_dense(x, units=1L)),
                  specials_to_oz = specials_to_oz, 
                  data = data,
                  output_dim = output_dim,
                  automatic_oz_check = TRUE,
                  param_nr = 1,
                  controls = controls,
                  parsing_options = form_control())
  )
  ll <- loop_through_pfc_and_call_trafo(list(res1))
  expect_true(all(!is.null(sapply(ll, dim))))
  expect_is(ll, "list")
  ll <- loop_through_pfc_and_call_trafo(list(res1), data)
  expect_true(all(!is.null(sapply(ll, dim))))
  expect_is(ll, "list")
  
})

test_that("to_matrix", {
  
  data <- list(array_input = dataset_mnist()$train[[1]][1:100,,])
  expect_equal(to_matrix(data), data[[1]])
  data <- data.frame(a=1:100,b=1:100)
  expect_equal(to_matrix(data), as.matrix(data))
  data <- list(a=1:100,b=1:100)
  expect_equal(to_matrix(data), do.call("cbind", data))
  
})