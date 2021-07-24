context("main entry: deepregression")

# helper function to check object dims
expect_object_dims = function(mod, data, loc = NULL, scale = NULL) {
  dims = lapply(coef(mod), length)
  expect_equal(dims[[1]], loc)
  expect_equal(dims[[2]], scale)

  mod %>% fit(epochs=2L, verbose = FALSE, view_metrics = FALSE)

  dims = lapply(coef(mod), length)
  expect_equal(dims[[1]], loc)
  expect_equal(dims[[2]], scale)

  mean <- mod %>% fitted()
  expect_true(length(mean) == nrow(data))
  expect_true(is.numeric(mean))

  mean <- mod %>% sd(data)
  expect_true(length(mean) == nrow(data))
  expect_true(is.numeric(mean))
}

test_that("simple additive model", {
  n <- 1500
  deep_model <- function(x) x %>%
    layer_dense(units = 2L, activation = "relu", use_bias = FALSE) %>%
    layer_dense(units = 1L, activation = "linear")

  x <- runif(n) %>% as.matrix()
  true_mean_fun <- function(xx) sin(10 * apply(xx, 1, mean) + 1)

  for (i in c(1, 3, 50)) {
    data = data.frame(matrix(x, ncol=i))
    if (ncol(data) == 1L) colnames(data) = "X1"
    y <- true_mean_fun(data)
    mod <- deepregression(
      y = y,
      data = data,
      list_of_formulae = list(loc = ~ 1 + d(X1), scale = ~1),
      list_of_deep_models = list(d = deep_model)
    )
    expect_object_dims(mod, data, 1, 1)
  }

  # 2 deep 1 structured + intercept
  data = data.frame(matrix(x, ncol=3))
  y <- true_mean_fun(data)
  mod <- deepregression(
    y = y,
    data = data,
    list_of_formulae = list(loc = ~ X3 + d(X1) + g(X2), scale = ~1),
    list_of_deep_models = list(d = deep_model, g = deep_model)
  )
  expect_object_dims(mod, data, 2, 1)


  # 2 deep 1 structured no intercept
  mod <- deepregression(
    y = y,
    data = data,
    list_of_formulae = list(loc = ~ -1 + X3 + d(X1) + g(X2), scale = ~1),
    list_of_deep_models = list(d = deep_model, g = deep_model)
  )
  expect_object_dims(mod, data, 1, 1)
})


test_that("generalized additive model", {
  n <- 1500
  deep_model <- function(x) x %>%
    layer_dense(units = 2L, activation = "relu", use_bias = FALSE) %>%
    layer_dense(units = 1L, activation = "linear")

  x <- runif(n) %>% as.matrix()
  true_mean_fun <- function(xx) sin(10 * apply(xx, 1, mean) + 1)

  # 2 deep 1 spline + intercept
  data = data.frame(matrix(x, ncol=3))
  y <- true_mean_fun(data)
  mod <- deepregression(
    y = y,
    data = data,
    list_of_formulae = list(loc = ~ s(X3, bs = "ts") + s(X1, bs = "cr") + g(X2), scale = ~1),
    list_of_deep_models = list(d = deep_model, g = deep_model)
  )
  suppressMessages(
    suppressWarnings(expect_object_dims(mod, data, 19, 1))
  )

  # # 2 deep 1 structured no intercept
  mod <- deepregression(
    y = y,
    data = data,
    list_of_formulae = list(loc = ~ X1 + d(X1) + g(X2), scale = ~ -1 + s(X3, bs = "tp")),
    list_of_deep_models = list(d = deep_model, g = deep_model)
  )
  expect_object_dims(mod, data, 2, 9)
})


# @FLO: redundant?
# test_that("generalized additive model", {
#   n <- 1500
#   deep_model <- function(x) x %>%
#     layer_dense(units = 2L, activation = "relu", use_bias = FALSE) %>%
#     layer_dense(units = 1L, activation = "linear")
# 
#   x <- runif(n) %>% as.matrix()
#   true_mean_fun <- function(xx) sin(10 * apply(xx, 1, mean) + 1)
# 
#   # 2 deep 1 spline + intercept
#   data = data.frame(matrix(x, ncol=3))
#   y <- true_mean_fun(data)
#   mod <- deepregression(
#     y = y,
#     data = data,
#     list_of_formulae = list(loc = ~ s(X3, bs = "ts") + s(X1, bs = "cr") + g(X2), scale = ~1),
#     list_of_deep_models = list(d = deep_model, g = deep_model)
#   )
#   suppressWarnings(expect_object_dims(mod, data, 19, 1))
# 
# 
#   # 2 deep 1 structured no intercept
#   data = cbind(data, X2 = runif(n), X3 = runif(n))
#   mod <- deepregression(
#     y = y,
#     data = data,
#     list_of_formulae = list(loc = ~ X1 + d(X1) + g(X2), scale = ~ -1 + s(X3, "tp")),
#     list_of_deep_models = list(d = deep_model, g = deep_model)
#   )
#   expect_object_dims(mod, data, 2, 9)
# })

test_that("deep generalized additive model with LSS", {
  set.seed(24)
  # generate the data
  n <- 500
  # training data; predictor
  x <- runif(n) %>% as.matrix()
  z <- runif(n) %>% as.matrix()
  true_mean_fun <- function(xx,zz) sin(10*xx) + zz^2
  true_sd_fun <- function(xl) exp(2 * xl)
  true_dgp_fun <- function(xx,zz)
  {
    eps <- rnorm(n) * true_sd_fun(xx)
    y <- true_mean_fun(xx, zz) + eps
    return(y)
  }
  y <- true_dgp_fun(x,z)
  data = data.frame(x = x, z = z)

  deep_model <- function(x) x %>%
    layer_dense(units = 2L, activation = "relu", use_bias = FALSE) %>%
    layer_dense(units = 1L, activation = "linear")

  mod <- deepregression(
    y = y,
    data = data,
    list_of_formulae = list(loc = ~ 1 + s(x, bs="tp") + d(z), scale = ~ 1 + x),
    list_of_deep_models = list(d = deep_model),
    family = "normal"
  )
  suppressWarnings(expect_object_dims(mod, data, 10,2))

})