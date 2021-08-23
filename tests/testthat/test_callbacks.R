context("Processors")

test_that("lin_processor", {
  
  data = data.frame(a=rnorm(2), b=rnorm(2), c=rnorm(2))
  term="lin(a + b + c)"
  expect_equal(lin_processor(term, data, 1, 1)$input_dim, 3)
  term="lin(1 + b + c)"
  expect_equal(lin_processor(term, data, 1, 1)$input_dim, 3)
  term="lin(a, b, c)"
  expect_equal(lin_processor(term, data, 1, 1)$input_dim, 3)
  term="lin(1, b, c)" # intercept is treated separately
  expect_equal(lin_processor(term, data, 1, 1)$input_dim, 2)
  
})

test_that("vc_processor", {
  
  n <- 40
  data = data.frame(a=rnorm(n), b=rnorm(n), c=rnorm(n), d=rnorm(n))
  term="vc(te(a,b), by=c(c,d))"
  expect_equal(vc_processor(term, data, 1, 1)$input_dim, 3)
  term="vc(te(a,b), by=c)"
  expect_equal(vc_processor(term, data, 1, 1)$input_dim, 3)
  term="vc(s(a), by=c(c,d))"
  expect_equal(vc_processor(term, data, 1, 1)$input_dim, 3)
  term="vc(s(a), by=c)" 
  expect_equal(vc_processor(term, data, 1, 1)$input_dim, 2)
  
})


test_that("processor", {
  
  form = ~ 1 + d(x) + s(x) + lasso(z) + ridge(z) + te(y) %OZ% (y + s(x)) + d(z) %OZ% s(x) + u
  data = data.frame(x = rnorm(100), y = rnorm(100), z = rnorm(100), u = rnorm(100))
  controls = smooth_control()
  output_dim = 1L
  param_nr = 1L
  d = dnn_placeholder_processor(function(x) layer_dense(x, units=1L))
  specials = c("s", "te", "ti", "vc", "lasso", "ridge", "offset", "vi", "fm", "vfm")
  specials_to_oz = c("d")
  

  res1 <- processor(form = form, 
                    d = dnn_placeholder_processor(function(x) layer_dense(x, units=1L)),
                    specials_to_oz = specials_to_oz, 
                    data = data,
                    output_dim = output_dim,
                    automatic_oz_check = TRUE,
                    param_nr = 1,
                    controls = controls)
  expect_is(res1, "list")
  expect_equal(length(res1), 9)
  expect_equal(sapply(res1, "[[", "nr"), 1:9)
  expect_type(sapply(res1, "[[", "input_dim"), "integer")
  
})

test_that("extractlen", {
  
  expect_equal(extractlen("a + b", data=list(a=rnorm(3), b=rnorm(3))),2)
  expect_equal(extractlen("something(a, b)", data=list(a=rnorm(3), b=rnorm(3))),2)
  expect_equal(extractlen("a + b + c", data=data.frame(a=rnorm(3), 
                                                       b=rnorm(3),
                                                       c=rnorm(3))),3)
  expect_equal(extractlen("s(a + b)", data=list(a=rnorm(3), b=rnorm(3))),2)
  
})

test_that("extractval", {
  
  expect_equal(extractval("lasso(x, la=1)", "la"), 1)
  expect_equal(extractval("lasso(x, abcd=1000)", "abcd"), 1000)
  expect_equal(extractval("lasso(x, y, z, u, abcd=1000)", "abcd"), 1000)
  expect_equal(extractval("lasso(x, y=1, z=2, u=3, abcd=1000)", "abcd"), 1000)
  expect_equal(extractval("lasso(x + y + z, abcd=1000)", "abcd"), 1000)
  
})