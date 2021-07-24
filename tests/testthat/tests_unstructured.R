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
      list_of_formulae = list(loc = as.formula(form), scale = ~1),
      list_of_deep_models = list(deep_model),
      cv_folds = 2
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
    "~ 1 + d(x) %OZ% s(z,by=fac)",
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
      list_of_formulae = list(loc = as.formula(form), scale = ~1),
      list_of_deep_models = list(deep_model)
    ))

    suppressWarnings(mod %>% fit(epochs=1, verbose = FALSE, view_metrics = FALSE))

    expect_is(mod, "deepregression")
    expect_true(!any(is.nan(unlist(coef(mod)))))

    suppressWarnings(res <- mod %>% predict(data))
    expect_true(is.numeric(res))
    expect_true(!any(is.nan(res)))
  }
})

test_that("array inputs", {
  mnist <- dataset_mnist()

  train_X <- list(x=array(mnist$train$x,
                          # so that we can use 2d conv later
                          c(dim(mnist$train$x),1))
  )
  subset <- 1:200
  train_X[[1]]<- train_X[[1]][subset,,,,drop=FALSE]
  train_y <- to_categorical(mnist$train$y[subset])

  conv_mod <- function(x) x %>%
    layer_conv_2d(filters = 16, kernel_size = c(3,3),
                  activation= "relu",
                  input_shape = shape(NULL, NULL, 1)) %>%
    layer_global_average_pooling_2d() %>%
    layer_dense(units = 10)

  simple_mod <- function(x) x %>%
    layer_dense(units = 4, activation = "relu") %>%
    layer_dense(units = 1, activation = "linear")

  z <- rnorm(length(subset))
  fac <- gl(4, length(subset)/4)
  m <- runif(length(z))

  list_as_input <- append(train_X, (data.frame(z=z, fac=fac, m=m)))

  mod <- deepregression(y = train_y, list_of_formulae =
                          list(logit = ~ 1 + simple_mod(z) + fac + conv_mod(x)),
                        data = list_as_input,
                        list_of_deep_models = list(simple_mod = simple_mod,
                                                   conv_mod = conv_mod),
                        family = "multinoulli")

  cvres <- mod %>% cv(epochs = 2, cv_folds = 2, batch_size=100)

  expect_is(cvres, "drCV")
  lapply(cvres, function(x) {
    expect_true(is.numeric(x$metrics$loss))
    expect_true(is.numeric(x$metrics$val_loss))
    expect_true(!any(is.nan(x$metrics$loss)))
  })

  expect_equal(dim(coef(mod)[[1]]), c(4, 10))
  mod %>% fit(epochs = 2,
              batch_size=100,
              view_metrics=FALSE,
              validation_split = NULL)
  expect_is(mod, "deepregression")
  expect_true(!any(is.nan(unlist(coef(mod)))))
})

context("Deep Specification")

test_that("deep specification", {
  set.seed(24)
  n <- 200
  b0 <- 1
  x <- runif(n) %>% as.matrix()
  z <- runif(n)
  fac <- gl(10, n/10)
  true_mean_fun <- function(xx) sin(10*xx) + b0
  # training data
  y <- true_mean_fun(x) + rnorm(n = n, mean = 0, sd = 2)
  k <- rnorm(length(x))
  data = data.frame(x = x, fac = fac, z = z)
  data$k <- k

  deep_model <- function(x) x %>%
    layer_dense(units = 4, activation = "relu") %>%
    layer_dense(units = 1, activation = "linear")

  another_deep_model <- function(x) x %>%
    layer_dense(units = 4, activation = "relu") %>%
    layer_dense(units = 1, activation = "linear")

  third_model <- function(x) x %>%
    layer_dense(units = 4, activation = "relu") %>%
    layer_dense(units = 1, activation = "linear")

  # works across different fomulae specifications
  formulae <- c(
    "~ d(x,z) + k",
    "~ d(x,z,k)",
    "~ d(x) + d(z)",
    "~ deep_model(x) + another_deep_model(z)",
    "~ deep_model(x,z) + another_deep_model(k)",
    "~ deep_model(x) + another_deep_model(z) + third_model(k)"
  )

  list_models <- list(deep_model = deep_model,
                      another_deep_model = another_deep_model,
                      third_model = third_model)
  list_models_wo_name <- list(deep_model, another_deep_model)
  use <- list(1,1,1:2,1:2,1:2,1:3)

  for (i in seq_len(length(formulae))) {
    form <- formulae[i]
    usei <- use[[i]]
    this_list <- list_models[usei]
    if (i %in% 1:3) {
      use_list <- list_models_wo_name[use[[i]]]
    } else {
      use_list <- list_models[use[[i]]]
    }
    suppressWarnings(
      mod <- deepregression(
        y = y,
        data = data,
        # define how parameters should be modeled
        list_of_formulae = list(loc = as.formula(form), scale = ~1),
        list_of_deep_models = use_list
      )
    )

    suppressWarnings(
      res <- mod %>% fit(epochs=2, verbose = FALSE, view_metrics = FALSE)
    )
    expect_is(mod, "deepregression")
    expect_true(!any(is.nan(unlist(coef(mod)))))
    expect_true(!any(is.nan(fitted(mod))))

    suppressWarnings(res <- mod %>% predict(data))
    expect_true(is.numeric(res))
    expect_true(!any(is.nan(res)))
  }
})


test_that("tffuns", {
  x = 2
  y = 3
  expect_is(tfe(x), "tensorflow.tensor")
  expect_is(tfsig(x), "tensorflow.tensor")
  expect_is(tfsoft(c(x,y)), "tensorflow.tensor")
  expect_is(tfsqrt(x), "tensorflow.tensor")
  expect_is(tfsq(x), "tensorflow.tensor")
  expect_is(tfdiv(x,y), c("numeric","tensorflow.tensor"))
  expect_is(tfrec(x), "tensorflow.tensor")
  expect_is(tfmult(x,y), "tensorflow.tensor")
})
