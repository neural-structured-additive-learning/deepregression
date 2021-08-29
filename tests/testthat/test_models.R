context("Model Builder")

test_that("model_builder", {
  
  n <- 1000
  data = data.frame(matrix(rnorm(4*n), c(n,4)))
  colnames(data) <- c("x1","x2","x3","xa")
  formula <- ~ 1 + deep_model(x1,x2,x3) + s(xa) + x1
  
  deep_model <- function(x) x %>%
    layer_dense(units = 32, activation = "relu", use_bias = FALSE) %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 8, activation = "relu") %>%
    layer_dense(units = 1, activation = "linear")
  
  y <- rnorm(n) + data$xa^2 + data$x1
  
  # check fake custom model
  mod <- deepregression(
    list_of_formulas = list(loc = ~ 1 + s(xa) + x1, scale = ~ 1),
    data = data, y = y,
    list_of_deep_models = list(deep_model = deep_model), 
    model_fun = build_customKeras()
  )
  expect_equal(class(mod$model)[1], "models.custom_train_step.customKeras")
  ret <- mod %>% fit(epochs = 2)
  
})