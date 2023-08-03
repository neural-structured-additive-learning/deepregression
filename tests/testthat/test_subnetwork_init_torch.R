context("Subnetwork Initilization")

test_that("subnetwork_init", {
  
  form = ~ 1 + d(x)  + ridge(z) + d(w) + u
  data = data.frame(x = rnorm(100), y = rnorm(100), 
                    z = rnorm(100), u = rnorm(100),
                    w = rnorm(100))
  controls = penalty_control()
  controls$with_layer <- TRUE
  controls$weight_options$warmstarts <- list("s(x)" = c(-4:4))
  controls$weight_options$general <- weight_control()[[1]]$general
  output_dim = 1L
  param_nr = 1L
  d = dnn_placeholder_processor(function() nn_sequential( 
                                  nn_linear(in_features = 1,
                                            out_features = 5),
                                  nn_linear(in_features = 5,
                                            out_features = 1)
                                  )
                                )
  specials = c("s", "te", "ti", "vc", "lasso", "ridge", "offset", "vi", "fm", "vfm")
  specials_to_oz = c("d")
  controls$gamdata <- precalc_gam(list(form), data, controls)
  
  pp <- suppressWarnings(
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
  
  res <- subnetwork_init_torch(list(pp))
  expect_true("nn_module" %in% class(res()))
  # does not work -- depending on the platform and tf version: 
  # expect_equal(c(unlist(sapply(res[[1]], function(x) x$shape[2]))),
  #              c(1, 9, rep(1, 7)))
  
})

