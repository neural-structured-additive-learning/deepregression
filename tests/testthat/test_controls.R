context("Controls")

test_that("penalty_control", {
  
  sc = penalty_control()
  data = data.frame(x = rnorm(100), y = rnorm(100))
  expect_equal(sc$sp_scale(data), 1/nrow(data))
  expect_false(sc$null_space_penalty)
  expect_false(sc$hat1)
  expect_true(sc$anisotropic)
  expect_true(sc$zero_constraint_for_smooths)
  expect_is(sc$df, "numeric")
  
  evaluated_smooth <- suppressWarnings(sc$defaultSmoothing(
    smoothCon(s(x),
              data=data,
              absorb.cons = sc$absorb_cons,
              null.space.penalty = sc$null_space_penalty
    ), df=6))
  expect_is(evaluated_smooth, "numeric")
  
})

test_that("orthog_control", {
  
  oc = orthog_control()
  expect_true(oc$orthogonalize)
  expect_identical(oc$orthog_type, "tf")
  
  dummy_fun <- function(x) x %>% layer_dense(5) %>% layer_dense(1)
  splitted_fun <- oc$split_fun(dummy_fun)
  expect_equal(as.character(body(splitted_fun[[1]])), 
               c("%>%", "x", "layer_dense(5)"))
  expect_equal(as.character(body(splitted_fun[[2]])), 
               c("%>%", "x", "layer_dense(1)"))
  
  expect_true(is.null(oc$deep_top))
  expect_equal(oc$orthog_fun, orthog_tf)
  
})
