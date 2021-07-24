context("get_contents")

test_that("get_contents", {
  nms = c("a", "b", "c", "x", "")
  # Set up a grid of formulas to test.
  gr = expand.grid(
    c("1", ""), nms, nms,
    c(paste0(sample(c("s(", "te(", "ti("), 1), sample(nms[-5], 1), ")"), ""),
    c(paste0(sample(c("s(", "te(", "ti("), 1), sample(nms[-5], 1), ")"), ""),
    sample(c("d(x)", "g(x)", ""), 1),
    sample(c("d(x)", "g(x)", ""), 1)
  )
  forms = apply(gr, 1, function(x) {
    x = Filter(function(x) x != "", unique(x))
    x = paste0("~", paste(x, collapse=" + "))
    if (x == "~") x = "~1"
    as.formula(x)
  })

  data = as.list(
    data.frame(a = runif(50), b = rnorm(50), c = 1:50, x = rnorm(50), k = 1)
  )
  networks = list(
    d = function(x) x %>% layer_dense(units = 1L, activation = "linear"),
    g = function(x) x %>% layer_dense(units = 1L, activation = "linear")
  )

  for (form in forms) {
    con = suppressWarnings(
      get_contents(
      lf = form,
      data = data,
      df = 1,
      defaultSmoothing = NULL,
      variable_names = names(data),
      network_names = names(networks)
    ))
    expect_is(con, "list")
    att = attributes(con)
    expect_equal(att$formula, form)
  }

  #expect_warning(
  expect_error(
    get_contents(
      lf = as.formula("~te(k)"),
      data = data,
      df = 1,
      defaultSmoothing = NULL,
      variable_names = names(data),
      network_names = names(networks)
    ), "reduce k") #, "2-dimensional")
})
