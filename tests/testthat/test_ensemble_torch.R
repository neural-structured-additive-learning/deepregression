context("Deep ensembles Torch")

test_that("deep ensemble", {

  set.seed(42)
  n <- 1000
  data <- data.frame(matrix(rnorm(4*n), c(n,4)))
  colnames(data) <- c("x1", "x2", "x3", "xa")

  y <- rnorm(n) + data$xa^2 + data$x1

  # check fake custom model
  mod <- deepregression(
    list_of_formulas = list(loc = ~ 1 + s(xa) + x1, scale = ~ 1),
    data = data, y = y, engine = "torch",
    orthog_options = orthog_control(orthogonalize = F)
  )

  cf_init <- coef(mod)
  ret <- ensemble(mod, epochs = 10, save_weights = TRUE, verbose = TRUE,
                  n_ensemble = 3)
  cf_post <- coef(mod)

  expect_identical(cf_init, cf_post)

  expect_length(cf <- coef.drEnsemble(ret), 3L)
  expect_equal(dim(cf$x1), c(1, 3))

  fitt <- fitted.drEnsemble(ret)
  expect_is(fitt, "list")

})

test_that("reinitializing weights", {

  n <- 100
  data = data.frame(matrix(rnorm(4*n), c(n,4)))
  colnames(data) <- c("x1","x2","x3","xa")
  formula <- ~ 1 + deep_model(x1,x2,x3) + s(xa) + x1
  
  deep_model <- function(){
    nn_sequential(
      nn_linear(in_features = 3, out_features = 32, bias = F),
      nn_relu(),
      nn_dropout(p = 0.2),
      nn_linear(in_features = 32, out_features = 8),
      nn_relu(),
      nn_linear(in_features = 8, out_features = 1)
    )
  }

  y <- rnorm(n) + data$xa^2 + data$x1

  # check fake custom model
  mod <- deepregression(
    list_of_formulas = list(loc = ~ 1 + s(xa) + x1, scale = ~ 1),
    data = data, y = y, engine = "torch",
    orthog_options = orthog_control(orthogonalize = F),
    list_of_deep_models = list(deep_model = deep_model),
    weight_options = weight_control(
      warmstart_weights =  list(list("x1" = 0), list())
    )
  )

  reinit_weights(mod, seed = 1)
  cfa <- coef(mod)
  
  reinit_weights(mod, seed = 2)
  cfb <- coef(mod)
  
  expect_false(all(cfa[[1]] == cfb[[1]]))

  fit(mod, epochs = 2)
  reinit_weights(mod, seed = 3)
  fit(mod, epochs = 2)

  expect_identical(cfa$x1, cfb$x1)
  })
