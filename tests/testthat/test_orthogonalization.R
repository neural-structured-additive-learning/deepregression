context("Orthogonalization")

test_that("orthogonalization", {
  set.seed(24)
  
  n <- 150
  ps <- c(1,3,5)
  b0 <- 1
  simnr <- 10
  true_sd <- 2
  
  deep_model <- function(x) x %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "linear")
  
  list_of_funs <-  list(function(x) sin(10*x),
                        function(x) tanh(3*x),
                        function(x) x^2,
                        function(x) cos(x*3-2)*(-x*3),
                        function(x) exp(x*2) - 1
  )
  
  for (p in 1:5) {
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
    
    cat("Fitting model with ", p, "orthogonalization(s) ... ")
    suppressWarnings(
      mod <- deepregression(
        y = y,
        data = data,
        list_of_formulas = list(loc = as.formula(form), scale = ~1),
        list_of_deep_models = list(d = deep_model)
      )
    )
    mod %>% fit(epochs=1, verbose = FALSE, view_metrics = FALSE)
    expect_is(mod, "deepregression")
    expect_true(!any(is.nan(unlist(coef(mod)))))
    expect_true(!any(is.nan(fitted(mod))))
    
    suppressWarnings(res <- mod %>% predict(data))
    expect_true(is.numeric(res))
    expect_true(!any(is.nan(res)))
  }
})

test_that("custom orthogonalization", {
  set.seed(24)
  n <- 500
  b0 <- 1
  x <- runif(n) %>% as.matrix()
  z <- runif(n)
  fac <- gl(10, n/10)
  true_mean_fun <- function(xx) sin(10*xx) + b0
  # training data
  y <- true_mean_fun(x) + rnorm(n = n, mean = 0, sd = 2)
  data = data.frame(x = x, fac = fac, z = z)
  
  deep_model <- function(x) x %>%
    layer_dense(units = 4, activation = "relu") %>%
    layer_dense(units = 1, activation = "linear")
  
  
  # first without the need for orthogonalization
  formulae <- c(
    "~ 0 + x",
    "~ 1 + x",
    "~ 1 + x + z",
    "~ 0 + s(x)",
    "~ 1 + s(x)",
    "~ 1 + s(x) + s(z)",
    "~ 1 + te(x,z)",
    "~ 1 + d(x) + z",
    "~ 1 + d(x,z)",
    "~ 1 + d(x) + s(z)",
    "~ 1 + s(x) + fac",
    "~ 1 + d(x) + fac",
    "~ 1 + d(x) + s(z,by=fac)",
    "~ 1 + d(x,z) %OZ% z",
    "~ 1 + d(x,z) %OZ% s(z)",
    "~ 1 + d(x,z) %OZ% (x+s(z))",
    "~ 1 + d(x) %OZ% s(z, by=fac)",
    "~ 1 + d(x,z) %OZ% z + x",
    "~ 1 + d(x,z) %OZ% s(z) + x",
    "~ 1 + d(x,z) %OZ% (x+s(z)) + z",
    "~ 1 + d(x) %OZ% s(z,by=fac) + x"
  )
  
  for (form in formulae) {
    suppressWarnings(mod <- deepregression(
      y = y,
      data = data,
      # define how parameters should be modeled
      list_of_formulas = list(loc = as.formula(form), scale = ~1),
      list_of_deep_models = list(d = deep_model)
    ))
    
    suppressWarnings(mod %>% fit(epochs=1, verbose = FALSE, view_metrics = FALSE))
    
    expect_is(mod, "deepregression")
    expect_true(!any(is.nan(unlist(coef(mod)))))
    
    suppressWarnings(res <- mod %>% predict(data))
    expect_true(is.numeric(res))
    expect_true(!any(is.nan(res)))
  }
})