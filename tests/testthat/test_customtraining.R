context("Custom Training")

test_that("Load and fit with custom keras model", {

  n <- 1500
  deep_model <- function(x) x %>%
    layer_dense(units = 2L, activation = "relu", use_bias = FALSE) %>%
    layer_dense(units = 1L, activation = "linear")
  
  x <- runif(n) %>% as.matrix()
  true_mean_fun <- function(xx) sin(10 * apply(xx, 1, mean) + 1)
  
  data = data.frame(matrix(x, ncol=50))
  y <- true_mean_fun(data)
  mod <- deepregression(
    y = y,
    data = data,
    list_of_formulas = list(loc = ~ 1 + d(X1), scale = ~1),
    list_of_deep_models = list(d = deep_model),
    model_fun = build_customKeras()
  )
  
  expect_is(mod, "deepregression")
  expect_length(mod, 3)
  expect_true(length(setdiff(names(mod), 
                             c("model", "init_params", "fit_fun")
  ))==0)
  expect_is(mod$model, "models.custom_train_step.customKeras")
  
  mod %>% fit(epochs = 2)

  mod2 <- deepregression(
    y = y,
    data = data,
    list_of_formulas = list(loc = ~ 1 + d(X1), scale = ~1),
    list_of_deep_models = list(d = deep_model)
  )
  
  mod2 %>% fit(epochs = 2)
    
  p1 <- mod %>% fitted()
  p2 <- mod2 %>% fitted()

  expect_equal(p1, p2)

})


test_that("Use multiple optimizers", {
  
  ld1 <- layer_dense(units = 8) 
  ld2 <- layer_dense(units = 16)
  ld3 <- layer_dense(units = 32)
  
  inp <- layer_input(shape = 4, batch_shape = NULL)
  
  outp <- inp %>% 
    ld1 %>% 
    ld2 %>% 
    ld3
  
  model = keras_model(inputs = inp, outputs = outp)

  optimizers = list(
    tf$keras$optimizers$Adam(learning_rate=1e-4),
    tf$keras$optimizers$Adam(learning_rate=1e-2)
  )
  
  optimizers_and_layers = list(tuple(optimizers[[1]], model$layers[[2]]), 
                               tuple(optimizers[[2]], model$layers[3:4]))
  optimizer = multioptimizer(optimizers_and_layers)
  model %>% compile(optimizer=optimizer, loss="mse")
  
  model %>% fit(x=matrix(rnorm(10*4), ncol=4), y=1:10,
                view_metrics = FALSE, verbose = FALSE)

  ### now in deepregression
  
  n <- 1500
  deep_model <- function(x) x %>%
    layer_dense(units = 2L, activation = "relu", use_bias = FALSE) %>%
    layer_dense(units = 1L, activation = "linear")
  
  x <- runif(n) %>% as.matrix()
  true_mean_fun <- function(xx) sin(10 * apply(xx, 1, mean) + 1)
  
  data = data.frame(matrix(x, ncol=50))
  y <- true_mean_fun(data)
  
  optimizer <- function(model){
    optimizers_and_layers = list(tuple(optimizers[[1]], model$layers[c(2,4)]), 
                                 tuple(optimizers[[2]], model$layers[c(5,8)]))
    optimizer = multioptimizer(optimizers_and_layers)
    return(optimizer)
  }
  
  mod <- deepregression(
    y = y,
    data = data,
    list_of_formulas = list(loc = ~ 1 + d(X1), scale = ~1),
    list_of_deep_models = list(d = deep_model),
    optimizer = optimizer
  )
  
  expect_is(mod, "deepregression")
  expect_length(mod, 3)
  expect_true(length(setdiff(names(mod), 
                             c("model", "init_params", "fit_fun")
  ))==0)

  mod %>% fit(epochs = 2)
  
  mod2 <- deepregression(
    y = y,
    data = data,
    list_of_formulas = list(loc = ~ 1 + d(X1), scale = ~1),
    list_of_deep_models = list(d = deep_model)
  )

  mod2 %>% fit(epochs = 2)
  
  expect_false(all((mod %>% fitted())==(mod2 %>% fitted())))
  
})
