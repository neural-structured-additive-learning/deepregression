context("deepregression layers")

test_that("custom layers", {
  
  set.seed(42)
  tf$random$set_seed(42)
  n <- 1000
  p <- 50
  x <- runif(n*p) %>% matrix(ncol=p)
  y <- 10*x[,1] + rnorm(n)
  
  inp <- layer_input(shape=c(p), 
                     batch_shape = NULL)
  out <- layer_hadamard_diff(units=1L, la=1, 
                             initu = tf$initializers$constant(1e-6),
                             initv = tf$initializers$constant(1e-6))(
                               inp
                             )
  mod <- keras_model(inp, out)
  
  mod %>% compile(loss="mse",
                  optimizer = tf$optimizers$Adam(learning_rate=1e-2))
  
  mod %>% fit(x=x, y=matrix(y), epochs=500L, 
              verbose=FALSE,
              validation_split=0.1)
  u <- as.matrix(mod$weights[[1]])
  v <- as.matrix(mod$weights[[2]])
  beta <- u^2-v^2
  expect_true(beta[1]>0.25)
  # does not work:
  expect_true(max(abs(beta[-1]))<1e-4)
  
})


test_that("lasso layers", {
  
  set.seed(42)
  tf$random$set_seed(42)
  n <- 1000
  p <- 50
  x <- runif(n*p) %>% matrix(ncol=p)
  y <- 10*x[,1] + rnorm(n)
  
  inp <- layer_input(shape=c(p), 
                     batch_shape = NULL)
  out <- tib_layer(units=1L, la=1)(inp)
  mod <- keras_model(inp, out)
  
  mod %>% compile(loss="mse",
                  optimizer = tf$optimizers$Adam(learning_rate=1e-2))
  
  mod %>% fit(x=x, y=matrix(y), epochs=500L, 
              verbose=FALSE,
              validation_split=0.1)
  expect_true(abs(Reduce("prod", lapply(mod$get_weights(),c))) < 1)
  
})
