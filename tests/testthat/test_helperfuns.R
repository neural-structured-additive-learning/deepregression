context("Helper Functions")

test_that("separate_define_relation", {
  
  form = ~ 1 + d(x) + s(x) + lasso(z) + ridge(z) + te(y) %OZ% (y + s(x)) + d(z) %OZ% s(x) 
  specials = c("s", "te", "ti", "vc", "lasso", "ridge", "offset", "vi", "fm", "vfm")
  specials_to_oz = c("d")
  
  # with automatic OZ
  res1 <- separate_define_relation(form, specials, specials_to_oz, automatic_oz_check = TRUE)
  expect_is(res1, "list")
  expect_equal(length(res1), 8)
  expect_equal(res1[[2]]$right_from_oz, c(1,5,6))
  # without automatic OZ
  res2 <- separate_define_relation(form, specials, specials_to_oz, automatic_oz_check = FALSE)
  expect_is(res2, "list")
  expect_equal(length(res2), 8)
  expect_equal(res2[[2]]$right_from_oz, c(5,6))
  # without DNN
  specials_to_oz = c()
  form = ~ 1 + s(x) + lasso(z) + ridge(z) + te(y) %OZ% (y + s(x))
  # with automatic OZ
  res3 <- separate_define_relation(form, specials, specials_to_oz, automatic_oz_check = TRUE)
  # without automatic OZ
  res4 <- separate_define_relation(form, specials, specials_to_oz, automatic_oz_check = FALSE)
  expect_equal(res3, res4)
  
})
  