context("deepregression methods")

test_that("all methods", {
  n <- 1500
  deep_model <- function(x) x %>%
    layer_dense(units = 2L, activation = "relu", use_bias = FALSE) %>%
    layer_dense(units = 1L, activation = "linear")

  x <- runif(n) %>% matrix(ncol=3)
  true_mean_fun <- function(xx) sin(10 * apply(xx, 1, mean) + 1) + 
    sapply(xx[, 3], function(x) rnorm(1, mean=0,sd=x))

  data = data.frame(matrix(x, ncol=3))
  y <- true_mean_fun(data)
  mod <- deepregression(
    y = y,
    data = data,
    list_of_formulas = list(loc = ~ X3 + d(X1) + g(X2), scale = ~X3),
    list_of_deep_models = list(d = deep_model, g = deep_model)
  )
  mod %>% fit(epochs=3L, verbose = FALSE, view_metrics = TRUE)

  mn = mean(mod, data)
  expect_is(mn, "matrix")
  expect_true(nrow(mn) == 500)
  expect_true(length(unique(mn)) > 1L)

  std = stddev(mod, data)
  expect_is(std, "matrix")
  expect_true(nrow(std) == 500)
  expect_true(length(unique(std)) > 1L)

  q95 = quant(mod, data, 0.95)
  q05 = quant(mod, data, 0.05)
  expect_true(all(q95 > mn  & mn > q05))

  expect_equal(predict(mod, data), mn)
  expect_equal(fitted(mod), mn)
  expect_true(plot(mod) == "No smooth effects. Nothing to plot.")

  cf = coef(mod)
  expect_is(cf, "list")
  # lapply(cf, function(x) {
  #   sl = x$structured_linear
  #   expect_is(sl, "matrix")
  #   expect_equal(dim(sl), c(2,1))
  #   NULL
  # })

  expect_output(print(mod), "Model")
  expect_output(print(mod), "Total params: 14")

  dst = get_distribution(mod)
  expect_is(dst, "python.builtin.object")

  ls = log_score(mod, data)
  expect_true(all(ls < 0))
  expect_true(all(dim(ls) == c(500, 1)))
})
