context("families")

test_that("tfd families", {
  families =  c("normal", "bernoulli", "bernoulli_prob", "beta", "betar",
    "cauchy", "chi2", "chi",
    "exponential", "gamma", "gammar",
    "gumbel", "half_cauchy", "half_normal", "horseshoe",
    "inverse_gamma", "inverse_gaussian", "laplace",
    "log_normal", "logistic", "multinomial", "multinoulli", "negbinom",
    "pareto_ls", "poisson", "poisson_lograte", "student_t", "student_t_ls",
    "uniform",
    "zip"
  )
  for (fam in families) {
    d = make_tfd_dist(fam)
    expect_is(d, "function")
    np = make_tfd_dist(fam, return_nrparams = TRUE)
    expect_true(np %in% c(1:3))
  }

  d = make_tfd_dist("zip", trafo_list = list(exp, exp))
  expect_is(d, "function")

  families = c("categorical",
   "dirichlet_multinomial",
   "dirichlet",
   "gamma_gamma",
   "geometric",
   "kumaraswamy",
   "truncated_normal",
   "von_mises",
   "von_mises_fisher",
   "wishart",
   "zipf",
   "binomial"
  )
  for (fam in families) {
    expect_error(make_tfd_dist(fam), "not implemented yet")
  }

})

test_that("tfd families can be fitted", {

  n <- 100

  # FIXME: Currently not working:
  # NA in fitted: "cauchy", "half_cauchy", "inverse_gamma", "student_t", "student_t_ls", "uniform"

  dists = c(
    "normal", "bernoulli", "bernoulli_prob",
    "beta", "betar", "chi2", "chi","exponential",
    "gamma", "gammar", "gumbel", "half_normal", "horseshoe",
    "inverse_gaussian", "laplace", "log_normal",
    "logistic", "negbinom", "negbinom",
    "pareto_ls", "poisson", "poisson_lograte"
  )

  for(dist in dists) {
    set.seed(24)
    x <- runif(n) %>% as.matrix()
    z <- runif(n) %>% as.matrix()
    y <- exp(as.matrix(0.5*x + rnorm(n, 0, 0.1*z) + 1))
    data = data.frame(x = x, z = z)
    if (dist %in% c("beta", "betar")) {
      y <- (y - min(y)) / (max(y) + 0.01 - min(y)) + runif(n, 1e-5, 1e-4)
    }
    suppressWarnings(
      mod <- deepregression(
        y = y,
        data = data,
        # define how parameters should be modeled
        list_of_formulae = list(~ 1 + x, ~ 1 + z, ~ 1),
        list_of_deep_models = NULL,
        family = dist, tf_seed = 44,
        optimizer = optimizer_rmsprop(lr = 0.000001)
      )
    )
    cat("Fitting", dist, "\n")
    res <- mod %>% fit(epochs=2L, verbose = FALSE, view_metrics = FALSE)
    expect_true(!sum(is.nan(unlist(res$metrics))) > 0)
    expect_true(!any(unlist(res$metrics)==Inf))
    expect_is(mod, "deepregression")
    expect_true(!any(is.nan(unlist(coef(mod)))))
    expect_true(!any(is.nan(fitted(mod))))
    suppressWarnings(res <- mod %>% predict(data))
    expect_true(is.numeric(res))
    expect_true(!any(is.nan(res)))
  }
})

test_that("mixdists", {
  mxdist = mix_dist_maker()
  expect_is(mxdist, "function")
  mkd = mxdist(matrix(rep(0.33, 12), ncol=12))
  expect_is(mkd, "python.builtin.object")
  expect_is(mkd$cdf, "python.builtin.method")
  expect_true(as.numeric(mkd$log_prob(1)) < 0)
})


test_that("multinorm", {
  mxdist = multinorm_maker()
  expect_is(mxdist, "function")
  mkd = mxdist(matrix(rep(0.33, 12), ncol=12))
  expect_is(mkd, "python.builtin.object")
  expect_is(mkd$cdf, "python.builtin.method")
  expect_true(as.numeric(mkd$log_prob(1:2)) < 0)
})

test_that("multinorm - no_cov", {
  mxdist = multinorm_maker(with_cov=FALSE)
  expect_is(mxdist, "function")
  mkd = mxdist(matrix(rep(1,4), nrow=1L))
  expect_is(mkd, "python.builtin.object")
  expect_is(mkd$cdf, "python.builtin.method")
  expect_true(as.numeric(mkd$log_prob(1:2)) < 0)
})

test_that("tfd_zip", {
  zipfun = tfd_zip(probs=c(0.1, 0.9), lambda=2)
  expect_is(zipfun, "python.builtin.object")
  expect_is(zipfun$cdf, "python.builtin.method")
  expect_true(as.numeric(zipfun$log_prob(1)) < 0)
  expect_true(as.numeric(zipfun$log_prob(0)) > as.numeric(zipfun$log_prob(1)))
})
