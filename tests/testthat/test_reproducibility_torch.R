context("reproducibility Torch")

test_that("reproducibility", {

  set.seed(24)

  # generate the data
  n <- 50
  b0 <- 1
  # training data; predictor
  x <- runif(n) %>% as.matrix()
  true_mean_fun <- function(xx) sin(10*xx) + b0
  # training data
  y <- true_mean_fun(x) + rnorm(n = n, mean = 0, sd = 2)
  data = data.frame(x = x)
  # test data
  x_test <- runif(n) %>% as.matrix()
  validation_data = data.frame(x = x_test)
  y_test <- true_mean_fun(x_test) + rnorm(n = n, sd = 2)

  deep_model <- function() nn_sequential(
    nn_linear(in_features = 1, out_features = 64, bias = F),
    nn_relu(),
    nn_linear(in_features = 64, out_features = 64),
    nn_relu(),
    nn_dropout(p = 0.2),
    nn_linear(in_features = 64, out_features = 16),
    nn_relu(),
    nn_linear(in_features = 16, out_features = 1)
  )
  

  mod1 <- deepregression(
    y = y,
    data = data,
    seed = 1L,
    list_of_formulas = list(loc = ~ 1 + d(x), scale = ~1),
    list_of_deep_models = list(d = deep_model), engine = "torch"
  )

  mod2 <- deepregression(
    y = y,
    data = data,
    seed = 1L,
    list_of_formulas = list(loc = ~ 1 + d(x), scale = ~1),
    list_of_deep_models = list(d = deep_model), engine = "torch"
  )
  mean1 <- mod1 %>% fitted()
  mean2 <- mod2 %>% fitted()
  # before training
  expect_equal(coef(mod1), coef(mod2))
  expect_equal(mean1, mean2)
  
  torch_manual_seed(1L)
  mod1 %>% fit(epochs=2L, verbose = FALSE, view_metrics = FALSE)
  mean1 <- mod1 %>% fitted()

  torch_manual_seed(1L)
  mod2 %>% fit(epochs=2L, verbose = FALSE, view_metrics = FALSE)
  mean2 <- mod2 %>% fitted()

  # after training
  expect_equal(coef(mod1), coef(mod2))
  expect_equal(mean1, mean2)
})
