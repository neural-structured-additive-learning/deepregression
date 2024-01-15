### Plan fürs testing: 

# Schritt eins: verschiedene Variablenwerte definieren
# Schritt 2: 
# eine reine Node-Formel erstellen
## dann auf node zugreifen über mod_node$model$get_layer(index = 3L)
### dann auf den ODT zugreifen und
#### a) die Dimensionen der Variablen testen
#### b) die Werte der Config testen 

#### Orthogonalization testen 
# gucken ob Warnung kommt. 

# # hierüber kann ich die prams bekommen 
# lay <- mod_node$model$get_layer("node_10")
# mod_node$model$get_layer(index = 3L)
# mod_node$model$layers
# lay_node <- lay$get_layer("oblivious_decision_tree_12")
# lay_node$get_config()
# ##### 
# 
# # hierüber bekomme ich die Dimensionen der Parameter
# lay_node$binary_lut$shape
# lay_node$log_temperatures
# lay_node$feature_selection_logits
# lay_node$feature_thresholds
# lay_node$log_temperatures
# lay_node$response
# 
# ### Hierüber die Config/User parameter bekommen: 
# lay_node$get_config()


context("Node")

test_that("node regression", {
  n <- 1000
  data_regr = data.frame(matrix(rnorm(4 * n), c(n, 4)))
  colnames(data_regr) <- c("x0", "x1", "x2", "x3")
  y_regr <- rnorm(n) + data_regr$x0^2 + data_regr$x1 + data_regr$x2*data_regr$x3 + data_regr$x2 + data_regr$x3
  
  tree_depth <- 5
  n_trees <- 2
  n_layers <- 3
  dim <- ncol(data_regr)
  units <- 1
  
  formula <- ~ node(x1, x2, x3, x0,  
                         n_trees = 2, 
                         n_layers = 3, 
                         tree_depth = 5)
  
  mod <- deepregression(
    list_of_formulas = list(loc = formula, scale = ~ 1),
    data = data_regr,
    y = y_regr
  )
  
  expect_is(mod, "deepregression")
  
  # Check values of parameters
  layer_node <- mod$model$get_layer(index = 2L)$get_layer(index = 1L)
  config <- layer_node$get_config() 
  #expect_true(c(config$n_trees, config$n_layers, config$tree_depth) == c(n_trees, n_layers, tree_depth))
  print(config)
  
  
  # dimensionen vor fit 
  # expect_equal evtl mit expect_true ersetzen
  expect_true(layer_node$binary_lut$shape == c(tree_depth, 2^tree_depth, 2))
  expect_true(layer_node$feature_selection_logits$shape == c(dim, n_trees, tree_depth))
  expect_true(layer_node$feature_thresholds$shape == c(n_trees, tree_depth))
  expect_true(layer_node$log_temperatures$shape == c(n_trees, tree_depth))
  expect_true(layer_node$response$shape == c(n_trees, units , 2^tree_depth))
  
  mod %>% fit()
  mod %>% predict()
  
  # dimensionen nach fit
  expect_true(layer_node$binary_lut$shape == c(tree_depth, 2^tree_depth, 2))
  expect_true(layer_node$feature_selection_logits$shape == c(dim, n_trees, tree_depth))
  expect_true(layer_node$feature_thresholds$shape == c(n_trees, tree_depth))
  expect_true(layer_node$log_temperatures$shape == c(n_trees, tree_depth))
  expect_true(layer_node$response$shape == c(n_trees, units , 2^tree_depth))

})

test_that("node bernoulli", {
  n <- 1000
  data = as.data.frame(matrix(rnorm(4 * n), c(n, 4)))
  colnames(data) <- c("x1", "x2", "x3", "x4")
  z <- -0.02*data$x1 + -0.3*data$x2 + -0.1*data$x3*data$x4
  pr <- 1/(1+exp(-z))
  y_tmp <- rbinom(n, 1 , pr)
  y <- to_categorical(y_tmp)
  
  tree_depth <- 5
  n_trees <- 2
  n_layers <- 3
  dim <- ncol(data)
  units <- 1

  formula <- ~ node(x1, x2, x3, x4,  
                    n_trees = 2, 
                    n_layers = 3, 
                    tree_depth = 5)
  
  mod <- deepregression(
    list_of_formulas = list(loc = formula),
    data = data,
    y = y,
    family = "bernoulli"
  )
  
  expect_is(mod, "deepregression")
  
  # Check values of parameters
  layer_node <- mod$model$get_layer(index = 2L)$get_layer(index = 1L)
  config <- layer_node$get_config() 
  #expect_true(c(config$n_trees, config$n_layers, config$tree_depth) == c(n_trees, n_layers, tree_depth))
  print(config)
  
  expect_true(layer_node$binary_lut$shape == c(tree_depth, 2^tree_depth, 2))
  expect_true(layer_node$feature_selection_logits$shape == c(dim, n_trees, tree_depth))
  expect_true(layer_node$feature_thresholds$shape == c(n_trees, tree_depth))
  expect_true(layer_node$log_temperatures$shape == c(n_trees, tree_depth))
  expect_true(layer_node$response$shape == c(n_trees, units , 2^tree_depth))
  
  mod %>% fit()
  mod %>% predict()
  
})

test_that("node multinoulli", {
  n <- 1000
  
  x1_0 <- rnorm(n*0.33, mean = 2, sd = 1)
  x2_0 <- rnorm(n*0.33, mean = 3, sd = 2)
  x_0 <- cbind(x1_0, x2_0, 0)
  
  x1_1 <- rnorm(n*0.33, mean = 7, sd = 1)
  x2_1 <- rnorm(n*0.33, mean = 9, sd = 2)
  x_1 <- cbind(x1_1, x2_1, 1)
  
  x1_2 <- rnorm(n*0.34, mean = 5, sd = 1)
  x2_2 <- rnorm(n*0.34, mean = 8, sd = 2)
  x_2 <- cbind(x1_2, x2_2, 2)
  
  data <- as.data.frame(rbind(x_0, x_1, x_2))
  colnames(data) <- c("x1", "x2", "y")
  y <- to_categorical(data$y)
  
  tree_depth <- 5
  n_trees <- 2
  n_layers <- 3
  dim <- 2
  units <- 3
  
  formula <- ~ node(x1, x2,  
                    n_trees = 2, 
                    n_layers = 3, 
                    tree_depth = 5)
  
  mod <- deepregression(
    list_of_formulas = list(loc = formula),
    data = data,
    y = y,
    family = "multinoulli"
  )
  
  # Check values of parameters
  layer_node <- mod$model$get_layer(index = 2L)$get_layer(index = 1L)
  config <- layer_node$get_config() 
  #expect_true(c(config$n_trees, config$n_layers, config$tree_depth) == c(n_trees, n_layers, tree_depth))
  print(config)
  
  expect_true(layer_node$binary_lut$shape == c(tree_depth, 2^tree_depth, 2))
  expect_true(layer_node$feature_selection_logits$shape == c(dim, n_trees, tree_depth))
  expect_true(layer_node$feature_thresholds$shape == c(n_trees, tree_depth))
  expect_true(layer_node$log_temperatures$shape == c(n_trees, tree_depth))
  expect_true(layer_node$response$shape == c(n_trees, units , 2^tree_depth))
  
  
  mod %>% fit()
  mod %>% predict()
  print(mod)

})

test_that("node overlap", {
  n <- 1000
  data_regr = data.frame(matrix(rnorm(4 * n), c(n, 4)))
  colnames(data_regr) <- c("x0", "x1", "x2", "x3")
  y_regr <- rnorm(n) + data_regr$x0^2 + data_regr$x1 + data_regr$x2*data_regr$x3 + data_regr$x2 + data_regr$x3

  formula <- ~ 1 + x0 + x1 + x2 +
    node(x2, n_trees = 2, n_layers = 2, tree_depth = 2) +
    node(x3, n_trees = 3, n_layers = 3, tree_depth = 3)

  expect_warning(deepregression(
    list_of_formulas = list(loc = formula, scale = ~ 1),
    data = data_regr,
    y = y_regr
  ))
})