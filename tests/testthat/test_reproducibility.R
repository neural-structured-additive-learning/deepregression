context("reproducibility")

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

  deep_model <- function(x) x %>%
    layer_dense(units = 64, activation = "relu", use_bias = FALSE) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "linear")

  mod1 <- deepregression(
    y = y,
    data = data,
    tf_seed = 1L,
    list_of_formulas = list(loc = ~ 1 + d(x), scale = ~1),
    list_of_deep_models = list(d = deep_model)
  )

  mod2 <- deepregression(
    y = y,
    data = data,
    tf_seed = 1L,
    list_of_formulas = list(loc = ~ 1 + d(x), scale = ~1),
    list_of_deep_models = list(d = deep_model)
  )
  mean1 <- mod1 %>% fitted()
  mean2 <- mod1 %>% fitted()
  # before training
  expect_equal(coef(mod1), coef(mod2))
  expect_equal(mean1, mean2)

  mod1 %>% fit(epochs=2L, verbose = FALSE, view_metrics = FALSE)
  mean1 <- mod1 %>% fitted()

  mod2 %>% fit(epochs=2L, verbose = FALSE, view_metrics = FALSE)
  mean2 <- mod1 %>% fitted()

  # after training
  expect_equal(coef(mod1), coef(mod2))
  expect_equal(mean1, mean2)
})
