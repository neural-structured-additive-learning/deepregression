context("Unstructured Data")

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

  mod <- deepregression(y = train_y, list_of_formulas =
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

  expect_equal(dim(coef(mod)[[1]]), c(3, 10))
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
      if(i==3) use_list <- use_list[1]
    } else {
      use_list <- list_models[use[[i]]]
    }
    suppressWarnings(
      mod <- deepregression(
        y = y,
        data = data,
        # define how parameters should be modeled
        list_of_formulas = list(loc = as.formula(form), scale = ~1),
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
