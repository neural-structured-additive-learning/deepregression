context("Orthogonalization torch")

test_that("orthogonalization not implemented", {
  set.seed(24)
  
  n <- 150
  b0 <- 1
  simnr <- 10
  true_sd <- 2
  
  p <- 1
  
  list_of_funs <-  list(function(x) sin(10*x),
                        function(x) tanh(3*x),
                        function(x) x^2,
                        function(x) cos(x*3-2)*(-x*3),
                        function(x) exp(x*2) - 1
  )
  
  X <- matrix(runif(p*n), ncol=p)
  partpred_l <- sapply(1:p, function(j) 4/j*X[,j])
  partpred_nl <- sapply(1:p, function(j)
    list_of_funs[[j]](X[,j]))
  
  true_mean <- b0 + rowSums(partpred_l) + rowSums(partpred_l)
  
  # training data
  y <- true_mean + rnorm(n = n, mean = 0, sd = true_sd)
  
  data = data.frame(X)
  colnames(data) <- paste0("V", 1:p)
  vars <- paste0("V", 1:p)
  form <- paste0("~ 1 + ", paste(vars, collapse = " + "), " + s(",
                 paste(vars, collapse = ") + s("), ") + d(",
                 paste(vars, collapse = ", "), ")")
  

  deep_model <- function() nn_sequential(
    nn_linear(1, 32),
    nn_relu(),
    nn_dropout(p = 0.2),
    nn_linear(32, 16),
    nn_relu(),
    nn_linear(16,1)
  )
  

  expect_error(
    deepregression(
    y = y,
    data = data,
    list_of_formulas = list(loc = as.formula(form), scale = ~1),
    list_of_deep_models = list(d = deep_model), engine = "torch"
    ), "Orthogonalization not implemented for torch"
  )

})
  