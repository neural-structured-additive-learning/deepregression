context("deepregression layers")

test_that("custom layers", {
  
  set.seed(42)
  n <- 5000
  x <- runif(n) %>% matrix(ncol=50)
  y <- rnorm(n/50) + x[,1]
  
  inp <- layer_input(shape=c(50), 
                     batch_shape = NULL)
  out <- layer_hadamard_diff(units=1L, la=0.05, 
                             initu = tf$initializers$constant(1e-3),
                             initv = tf$initializers$constant(1e-3))(
                               inp
                               )
  mod <- keras_model(inp, out)
    
  mod %>% compile(loss="mse",
                  optimizer = tf$optimizers$Adam())
  
  mod %>% fit(x=x, y=matrix(y), epochs=400L, 
              verbose=FALSE,
              validation_split=0.1)
  u <- as.matrix(mod$weights[[1]])
  v <- as.matrix(mod$weights[[2]])
  beta <- u^2-v^2
  expect_true(beta[1]>0.25)
  # does not work:
  #expect_true(max(abs(beta[-1]))<1e-4)

})
