context("Model Builder")

test_that("model_builder", {

  n <- 1000
  data = data.frame(matrix(rnorm(4*n), c(n,4)))
  colnames(data) <- c("x1","x2","x3","xa")
  formula <- ~ 1 + deep_model(x1,x2,x3) + s(xa) + x1

  deep_model <- function(x) x %>%
    layer_dense(units = 32, activation = "relu", use_bias = FALSE) %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 8, activation = "relu") %>%
    layer_dense(units = 1, activation = "linear")

  y <- rnorm(n) + data$xa^2 + data$x1

  # check fake custom model
  mod <- deepregression(
    list_of_formulas = list(loc = ~ 1 + s(xa) + x1, scale = ~ 1),
    data = data, y = y,
    list_of_deep_models = list(deep_model = deep_model),
    model_fun = build_customKeras()
  )

  expect_equal(class(mod$model)[1], "models.custom_train_step.customKeras")

  cf_init <- coef(mod)
  ret <- ensemble.deepregression(mod, epochs = 10, save_weights = TRUE)
  cf_post <- coef(mod)

  expect_identical(cf_init, cf_post)

  edist <- get_ensemble_distribution(ret)
  - mean(diag(tfd_log_prob(edist, y)$numpy()))

})

test_that("reinitializing weights", {

  n <- 100
  data = data.frame(matrix(rnorm(4*n), c(n,4)))
  colnames(data) <- c("x1","x2","x3","xa")
  formula <- ~ 1 + deep_model(x1,x2,x3) + s(xa) + x1

  deep_model <- function(x) x %>%
    layer_dense(units = 32, activation = "relu", use_bias = FALSE) %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 8, activation = "relu") %>%
    layer_dense(units = 1, activation = "linear")

  y <- rnorm(n) + data$xa^2 + data$x1

  # check fake custom model
  mod <- deepregression(
    list_of_formulas = list(loc = ~ 1 + s(xa) + x1, scale = ~ 1),
    data = data, y = y,
    list_of_deep_models = list(deep_model = deep_model),
    model_fun = build_customKeras(),
    weight_options = weight_control(
      warmstart_weights =  list(list("x1" = 0), list())
    )
  )

  cfa <- coef(mod)
  reinit_weights.deepregression(mod)
  cfb <- coef(mod)

  fit(mod, epochs = 2)
  reinit_weights.deepregression(mod)
  fit(mod, epochs = 2)

  expect_identical(cfa$x1, cfb$x1)

})
