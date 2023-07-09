context("Custom Training")


test_that("Use multiple optimizers torch", {
  
  fc1 <- nn_linear(1, 50)
  fc2 <- nn_linear(50, 1)
  net <- nn_module(
    "Net",
    initialize = function() {
      self$fc1 <- fc1
      self$fc2 <- fc2
    },
    forward = function(x) {
      x %>% 
        self$fc1() %>% 
        nnf_relu() %>% 
        self$fc2()
    }
  )
  
  set_optimizers = function(model_parameters, lr_fc1 = 0, lr_fc2 = 0.01) {
    list(
      opt_fc1 = optim_adam(model_parameters[1:2], lr = lr_fc1),
      opt_fc2 = optim_adam(model_parameters[3:4], lr = lr_fc2)
    )
  }
  
  pre_fit <- net %>% luz::setup(loss = nn_mse_loss(), optimizer = set_optimizers)
  
  pre_fit_weights <- lapply(pre_fit()$parameters, function(x) as_array(x))
  fit(pre_fit,
      data = list(
                  matrix(rnorm(100), ncol = 1),
                  matrix(rnorm(100), ncol = 1)),
      epochs = 10)
  
  after_fit_weights <- lapply(pre_fit()$parameters, function(x) as_array(x))
  expect_true(
    identical(pre_fit_weights[1:2],
              after_fit_weights[1:2])
  )
  expect_false(
    identical(pre_fit_weights[3:4],
              after_fit_weights[3:4]))
  
  ### now in deepregression
  
  net <- nn_module(
    "Net",
    initialize = function() {
      self$fc1 <- nn_linear(1, 50)
      self$fc2 <- nn_linear(50, 1)
    },
    forward = function(x) {
      x %>% 
        self$fc1() %>% 
        nnf_relu() %>% 
        self$fc2()
    }
  )
  
  
  
  n <- 1500
  x <- runif(n) %>% as.matrix()
  true_mean_fun <- function(xx) sin(10 * apply(xx, 1, mean) + 1)
  
  data = data.frame(matrix(x, ncol=50))
  y <- true_mean_fun(data)
  
  set_optimizers = function(model_parameters, 
                            lr_loc_fc1 = 0,
                            lr_loc_fc2 = 0.01,
                            lr_scale_inter = 0) {
    list(
      opt_loc_fc1 = optim_adam(model_parameters[1:2], lr = lr_loc_fc1),
      opt_loc_fc2 = optim_adam(model_parameters[3:4], lr = lr_loc_fc2),
      opt_scale_intercept = optim_adam(model_parameters[5], lr = lr_scale_inter)
    )
  }
  
  mod <- deepregression(
    y = y,
    data = data,
    list_of_formulas = list(loc = ~-1 + d(X1), scale = ~1),
    list_of_deep_models = list(d = net), engine = "torch",
    optimizer = set_optimizers
  )
  
  expect_is(mod, "deepregression")
  expect_length(mod, 4)
  expect_true(length(setdiff(names(mod), 
                             c("model", "init_params", "fit_fun", "engine")
  ))==0)
  
  pre_fit_weights <- get_weights_torch(mod)
  mod %>% fit(epochs = 10)
  after_fit_weights <- get_weights_torch(mod)
  
  expect_equal(
    pre_fit_weights[1:2],
    after_fit_weights[1:2]
  )
  expect_equal(
    pre_fit_weights[5],
    after_fit_weights[5]
  )
  
  expect_false(
    identical(
      list(pre_fit_weights[3:4]),
      list(after_fit_weights[3:4])
      )
    )
  
})
