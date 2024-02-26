context("Node")

test_that("node regression", {
  n <- 1000
  data_regr = data.frame(matrix(rnorm(4 * n), c(n, 4)))
  colnames(data_regr) <- c("x0", "x1", "x2", "x3")
  y_regr <-
    rnorm(n) + data_regr$x0 ^ 2 + data_regr$x1 + data_regr$x2 * data_regr$x3 + data_regr$x2 + data_regr$x3
  
  tree_depth <- 5
  n_trees <- 2
  n_layers <- 3
  dim <- ncol(data_regr)
  units <- 1
  
  formula <- ~ node(
    x1,
    x2,
    x3,
    x0,
    n_trees = 2,
    n_layers = 3,
    tree_depth = 5
  )
  
  mod <- deepregression(
    list_of_formulas = list(loc = formula, scale = ~ 1),
    data = data_regr,
    y = y_regr
  )
  
  expect_is(mod, "deepregression")
  
  # retrieve NODE-model and model/layer configs
  model_node <- mod$model$get_layer(index = 2L)
  layer_node <-
    mod$model$get_layer(index = 2L)$get_layer(index = 1L)
  config_node_layer <- layer_node$get_config()
  config_node_model <- model_node$get_config()
  
  # check hyperparameters
  expect_equal(
    c(
      config_node_model$n_trees,
      config_node_model$n_layers,
      config_node_model$tree_depth,
      config_node_model$units
    ),
    c(n_trees, n_layers, tree_depth, units)
  )
  expect_equal(
    c(
      config_node_layer$n_trees,
      config_node_layer$depth,
      config_node_layer$units
    ),
    c(n_trees, tree_depth, units)
  )
  
  # check dimensions of NODE-model/ODT-Layers
  expect_true(layer_node$binary_lut$shape == c(tree_depth, 2 ^ tree_depth, 2))
  expect_true(layer_node$feature_selection_logits$shape == c(dim, n_trees, tree_depth))
  expect_true(layer_node$feature_thresholds$shape == c(n_trees, tree_depth))
  expect_true(layer_node$log_temperatures$shape == c(n_trees, tree_depth))
  expect_true(layer_node$response$shape == c(n_trees, units , 2 ^ tree_depth))
  
  mod %>% fit()
  mod %>% predict()
  
  # check dimensions of NODE-model/ODT-Layers after fitting the model
  expect_true(layer_node$binary_lut$shape == c(tree_depth, 2 ^ tree_depth, 2))
  expect_true(layer_node$feature_selection_logits$shape == c(dim, n_trees, tree_depth))
  expect_true(layer_node$feature_thresholds$shape == c(n_trees, tree_depth))
  expect_true(layer_node$log_temperatures$shape == c(n_trees, tree_depth))
  expect_true(layer_node$response$shape == c(n_trees, units , 2 ^ tree_depth))
  
})

test_that("node bernoulli", {
  n <- 1000
  data = as.data.frame(matrix(rnorm(4 * n), c(n, 4)))
  colnames(data) <- c("x1", "x2", "x3", "x4")
  z <- -0.02 * data$x1+-0.3 * data$x2+-0.1 * data$x3 * data$x4
  pr <- 1 / (1 + exp(-z))
  y_tmp <- rbinom(n, 1 , pr)
  y <- to_categorical(y_tmp)
  
  tree_depth <- 5
  n_trees <- 2
  n_layers <- 3
  dim <- ncol(data)
  units <- 1
  
  formula <- ~ node(
    x1,
    x2,
    x3,
    x4,
    n_trees = 2,
    n_layers = 3,
    tree_depth = 5
  )
  
  mod <- deepregression(
    list_of_formulas = list(loc = formula),
    data = data,
    y = y,
    family = "bernoulli"
  )
  
  expect_is(mod, "deepregression")
  
  # retrieve NODE-model
  model_node <- mod$model$get_layer(index = 2L)
  layer_node <-
    mod$model$get_layer(index = 2L)$get_layer(index = 1L)
  config_node_layer <- layer_node$get_config()
  config_node_model <- model_node$get_config()
  
  # check hyperparameters
  expect_equal(
    c(
      config_node_model$n_trees,
      config_node_model$n_layers,
      config_node_model$tree_depth,
      config_node_model$units
    ),
    c(n_trees, n_layers, tree_depth, units)
  )
  expect_equal(
    c(
      config_node_layer$n_trees,
      config_node_layer$depth,
      config_node_layer$units
    ),
    c(n_trees, tree_depth, units)
  )
  
  
  # check dimensions of NODE-model/ODT-Layers
  expect_true(layer_node$binary_lut$shape == c(tree_depth, 2 ^ tree_depth, 2))
  expect_true(layer_node$feature_selection_logits$shape == c(dim, n_trees, tree_depth))
  expect_true(layer_node$feature_thresholds$shape == c(n_trees, tree_depth))
  expect_true(layer_node$log_temperatures$shape == c(n_trees, tree_depth))
  expect_true(layer_node$response$shape == c(n_trees, units , 2 ^ tree_depth))
  
  mod %>% fit()
  mod %>% predict()
  
  # check dimensions of NODE-model/ODT-Layers after fitting the model
  expect_true(layer_node$binary_lut$shape == c(tree_depth, 2 ^ tree_depth, 2))
  expect_true(layer_node$feature_selection_logits$shape == c(dim, n_trees, tree_depth))
  expect_true(layer_node$feature_thresholds$shape == c(n_trees, tree_depth))
  expect_true(layer_node$log_temperatures$shape == c(n_trees, tree_depth))
  expect_true(layer_node$response$shape == c(n_trees, units , 2 ^ tree_depth))
  
})

test_that("node multinoulli", {
  n <- 1000
  
  x1_0 <- rnorm(n * 0.33, mean = 2, sd = 1)
  x2_0 <- rnorm(n * 0.33, mean = 3, sd = 2)
  x_0 <- cbind(x1_0, x2_0, 0)
  
  x1_1 <- rnorm(n * 0.33, mean = 7, sd = 1)
  x2_1 <- rnorm(n * 0.33, mean = 9, sd = 2)
  x_1 <- cbind(x1_1, x2_1, 1)
  
  x1_2 <- rnorm(n * 0.34, mean = 5, sd = 1)
  x2_2 <- rnorm(n * 0.34, mean = 8, sd = 2)
  x_2 <- cbind(x1_2, x2_2, 2)
  
  data <- as.data.frame(rbind(x_0, x_1, x_2))
  colnames(data) <- c("x1", "x2", "y")
  y <- to_categorical(data$y)
  
  tree_depth <- 5
  n_trees <- 2
  n_layers <- 3
  dim <- 2
  units <- 3
  
  formula <- ~ node(x1,
                    x2,
                    n_trees = 2,
                    n_layers = 3,
                    tree_depth = 5)
  
  mod <- deepregression(
    list_of_formulas = list(loc = formula),
    data = data,
    y = y,
    family = "multinoulli"
  )
  
  # retrieve NODE-model
  model_node <- mod$model$get_layer(index = 2L)
  layer_node <-
    mod$model$get_layer(index = 2L)$get_layer(index = 1L)
  config_node_layer <- layer_node$get_config()
  config_node_model <- model_node$get_config()
  
  # check hyperparameters
  expect_equal(
    c(
      config_node_model$n_trees,
      config_node_model$n_layers,
      config_node_model$tree_depth,
      config_node_model$units
    ),
    c(n_trees, n_layers, tree_depth, units)
  )
  expect_equal(
    c(
      config_node_layer$n_trees,
      config_node_layer$depth,
      config_node_layer$units
    ),
    c(n_trees, tree_depth, units)
  )
  
  
  # check dimensions of NODE-model/ODT-Layers
  expect_true(layer_node$binary_lut$shape == c(tree_depth, 2 ^ tree_depth, 2))
  expect_true(layer_node$feature_selection_logits$shape == c(dim, n_trees, tree_depth))
  expect_true(layer_node$feature_thresholds$shape == c(n_trees, tree_depth))
  expect_true(layer_node$log_temperatures$shape == c(n_trees, tree_depth))
  expect_true(layer_node$response$shape == c(n_trees, units , 2 ^ tree_depth))
  
  
  mod %>% fit()
  mod %>% predict()
  print(mod)
  
  # check dimensions of NODE-model/ODT-Layers after fitting the model
  expect_true(layer_node$binary_lut$shape == c(tree_depth, 2 ^ tree_depth, 2))
  expect_true(layer_node$feature_selection_logits$shape == c(dim, n_trees, tree_depth))
  expect_true(layer_node$feature_thresholds$shape == c(n_trees, tree_depth))
  expect_true(layer_node$log_temperatures$shape == c(n_trees, tree_depth))
  expect_true(layer_node$response$shape == c(n_trees, units , 2 ^ tree_depth))
  
})

test_that("node overlap", {
  n <- 1000
  data_regr = data.frame(matrix(rnorm(4 * n), c(n, 4)))
  colnames(data_regr) <- c("x0", "x1", "x2", "x3")
  y_regr <- rnorm(n) + data_regr$x0 ^ 2 + data_regr$x1 +
    data_regr$x2 * data_regr$x3 + data_regr$x2 + data_regr$x3
  
  deep_model <- function(x) {
    x %>%
      layer_dense(units = 32,
                  activation = "relu",
                  use_bias = FALSE) %>%
      layer_dropout(rate = 0.2) %>%
      layer_dense(units = 8, activation = "relu") %>%
      layer_dense(units = 1, activation = "linear")
  }
  
  formula_ov_node_structured <- ~ 1 + x0 + x1 + x2 +
    node(x2,
         n_trees = 2,
         n_layers = 2,
         tree_depth = 2)
  
  formula_ov_node_structured_deep <- ~ 1 + x1 + x2 +
    node(x2,
         x3,
         n_trees = 2,
         n_layers = 2,
         tree_depth = 2) +
    deep_model(x0, x1, x2)
  
  expect_warning(deepregression(
    list_of_formulas = list(loc = formula_ov_node_structured, scale = ~ 1),
    data = data_regr,
    y = y_regr
  ))
  
  expect_warning(
    deepregression(
      list_of_formulas = list(loc = formula_ov_node_structured_deep, scale = ~ 1),
      data = data_regr,
      y = y_regr,
      list_of_deep_models = list(deep_model = deep_model)
    )
  )
})