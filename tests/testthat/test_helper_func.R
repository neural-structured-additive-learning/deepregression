context("various")

test_that("check_and_install", {
  expect_message(check_and_install(), "Tensorflow found, skipping tensorflow installation!")
})


test_that("callbacks can be instantiated", {
  cb = WeightHistory$new()
  expect_is(cb, "KerasCallback")
  cb = auc_roc$new(training = as.list(1:2), validation = as.list(1:2))
  expect_is(cb, "KerasCallback")
  cb = KerasMetricsCallback_custom$new()
  expect_is(cb, "KerasCallback")
})

