context("Main entry: deepregression")

test_that("Simple additive model", {
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
      list_of_formulas = list(loc = ~ 1 + d(X1), scale = ~1),
      list_of_deep_models = list(d = deep_model)
    )
    expect_is(mod, "deepregression")
    expect_length(mod, 3)
    expect_true(length(setdiff(names(mod), 
                               c("model", "init_params", "fit_fun")
                               )
                       )==0)
  }

  # 2 deep 1 structured + intercept
  data = data.frame(matrix(x, ncol=3))
  y <- true_mean_fun(data)
  mod <- deepregression(
    y = y,
    data = data,
    list_of_formulas = list(loc = ~ X3 + d(X1) + g(X2), scale = ~1),
    list_of_deep_models = list(d = deep_model, g = deep_model)
  )
  
  expect_is(mod, "deepregression")
  expect_length(mod, 3)
  expect_true(length(setdiff(names(mod), 
                             c("model", "init_params", "fit_fun")
  )
  )==0)


  # 2 deep 1 structured no intercept
  mod <- deepregression(
    y = y,
    data = data,
    list_of_formulas = list(loc = ~ -1 + X3 + d(X1) + g(X2), scale = ~1),
    list_of_deep_models = list(d = deep_model, g = deep_model)
  )
  expect_is(mod, "deepregression")
  expect_length(mod, 3)
  expect_true(length(setdiff(names(mod), 
                             c("model", "init_params", "fit_fun")
  )
  )==0)
  
  deep_model_2 <- function(x) x %>% 
    layer_dense(10) %>% 
    layer_dense(2)
  
  # shared deep
  mod <- deepregression(
    y = y,
    data = data,
    list_of_formulas = list(loc = ~ -1 + X3 + d(X1), 
                            scale = ~1 + d(X1), 
                            both = ~ g(X2)),
    list_of_deep_models = list(d = deep_model, g = deep_model_2),
    mapping = list(1,2,1:2),
    add_layer_shared_pred = function(x) x
  )
  
  expect_is(mod, "deepregression")
  
  mod %>% fit(epochs = 2)
  
})


test_that("Generalized additive model", {
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
    list_of_formulas = list(loc = ~ s(X3, bs = "ts") + s(X1, bs = "cr") + g(X2), scale = ~1),
    list_of_deep_models = list(d = deep_model, g = deep_model)
  )
  
  expect_is(mod, "deepregression")
  
  mod %>% fit(epochs = 2)

  # # 2 deep 1 structured no intercept
  mod <- deepregression(
    y = y,
    data = data,
    list_of_formulas = list(loc = ~ X1 + d(X1) + g(X2), scale = ~ -1 + s(X3, bs = "tp")),
    list_of_deep_models = list(d = deep_model, g = deep_model)
  )
  
  expect_is(mod, "deepregression")
  
  mod %>% fit(epochs = 2)
  
  pred <- mod %>% predict(data)
})


test_that("Deep generalized additive model with LSS", {
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
    list_of_formulas = list(loc = ~ 1 + s(x, bs="tp") + d(z), scale = ~ 1 + x),
    list_of_deep_models = list(d = deep_model),
    family = "normal"
  )
  
  expect_is(mod, "deepregression")
  
  mod %>% fit(epochs = 2)
})

test_that("Multivariate response", {
  
  n <- 100
  p <- 10
  
  x <- matrix(runif(n*p), ncol=p)
  
  data <- data.frame(x = x)
  
  y <- matrix(rnorm(n*3), ncol=3)
  mod <- deepregression(
    y = y,
    data = data,
    list_of_formulas = list(loc = ~ x.1 + x.2, scale = ~1)
  )
  
  expect_is(mod, "deepregression")
  
  mod %>% fit(epochs = 3L)
  
})

test_that("Generalized additive model with RWT in formula", {
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
    list_of_formulas = list(loc = ~ X2 %X% (s(X3, bs = "ts") + s(X1, bs = "cr")) + g(X2), scale = ~1),
    list_of_deep_models = list(d = deep_model, g = deep_model)
  )
  
  expect_is(mod, "deepregression")
  
  mod %>% fit(epochs = 2)
  
})


test_that("GAMs with shared weights", {
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
    list_of_formulas = list(loc = ~ s(X3) + s(X2), 
                            scale = ~1 + s(X1) + X2),
    list_of_deep_models = list(d = deep_model, g = deep_model),
    weight_options = weight_control(
      shared_layers = list(list(c("s(X3)","s(X2)")), NULL)
    )
  )

  expect_equal(mod$init_params$parsed_formulas_contents[[1]][[1]]$shared_name,
               mod$init_params$parsed_formulas_contents[[1]][[2]]$shared_name)
  
  expect_equal(c(coef(mod, which_param = 1)[[1]]),
               c(coef(mod, which_param = 1)[[2]]))

  mod %>% fit(epochs = 2)

  expect_equal(c(coef(mod, which_param = 1)[[1]]),
               c(coef(mod, which_param = 1)[[2]]))
  
})

test_that("GAMs with fixed weights", {
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
    list_of_formulas = list(loc = ~ s(X3) + g(X2), scale = ~1 + s(X1) + X2),
    list_of_deep_models = list(d = deep_model, g = deep_model),
    weight_options = weight_control(
      warmstart_weights = list(list("s(X3)" = -4:4), list("s(X1)" = rep(1,9), "X2" = 5))
    )
  )
  
  expect_is(mod, "deepregression")
  
  expect_equal(c(coef(mod, which_param = 1)[[1]]),
               -4:4)
  
  expect_equal(c(coef(mod, which_param = 2)[[1]]),
               rep(1,9))
  
  expect_equal(c(coef(mod, which_param = 2)[[2]]), 5)
  
  data = data.frame(matrix(x, ncol=3))
  y <- true_mean_fun(data)
  
  mod <- deepregression(
    y = y,
    data = data,
    list_of_formulas = list(loc = ~ s(X3) + g(X2), scale = ~1 + s(X1) + X2),
    list_of_deep_models = list(d = deep_model, g = deep_model),
    weight_options = weight_control(
      warmstart_weights = list(list("s(X3)" = -4:4), 
                               list("s(X1)" = rep(1,9), "X2" = 5)),
      specific_weight_options = list(list("s(X3)" = list(trainable = FALSE)),
                                     list("X2" = list(trainable = FALSE)))
    )
  )
  
  
  mod %>% fit(epochs = 2)
  
  expect_equal(c(coef(mod, which_param = 1)[[1]]),
               -4:4)
  
  # is not set to FALSE, so should be different
  expect_true(sum(abs(c(coef(mod, which_param = 2)[[1]])-rep(1,9)))>0)
  
  expect_equal(c(coef(mod, which_param = 2)[[2]]), 5)
  
  ### fix intercept (relevant for deeptrafo)
  
  mod <- deepregression(
    y = y,
    data = data,
    list_of_formulas = list(loc = ~ 1 + s(X3) + g(X2), scale = ~1 + s(X1) + X2),
    list_of_deep_models = list(d = deep_model, g = deep_model),
    weight_options = weight_control(
      warmstart_weights = list(list("1" = 0), NULL),
      specific_weight_options = list(list("1" = list(trainable = FALSE)),
                                     NULL)
    )
  )
  
  expect_is(mod, "deepregression")
  
  expect_equal(c(coef(mod, which_param = 1)$`(Intercept)`), 0)
  mod %>% fit(epochs = 2)
  expect_equal(c(coef(mod, which_param = 1)$`(Intercept)`), 0)
  
})
