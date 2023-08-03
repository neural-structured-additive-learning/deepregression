context("families Torch")

test_that("torch families", {
  families =  c("normal", "bernoulli", "bernoulli_prob", "gamma", 
                "poisson" )
  
  for (fam in families) {
    d = make_torch_dist(fam)
    expect_is(d, "function")
    np = attr(d, "nrparams_dist")
    expect_true(np %in% c(1:3))
  }

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
               "zipf", "beta",
               "betar",
               "cauchy",
               "chi2",
               "chi",
               "exponential",
               "gammar",
               "gumbel",
               "half_cauchy",
               "half_normal",
               "horseshoe",
               "inverse_gamma",
               "inverse_gaussian",
               "laplace",
               "log_normal",
               "logistic",
               "multinomial",
               "multinoulli",
               "negbinom",
               "pareto_ls",
               "poisson_lograte",
               "student_t",
               "student_t_ls",
               "uniform",
               "zip")
  for (fam in families) {
    expect_error(make_torch_dist(fam), "not implemented")
  }

})

test_that("torch families can be fitted", {

  n <- 100

  dists = c(
    "normal", "bernoulli", "bernoulli_prob",
    "gamma", "poisson")

  for(dist in dists) {
    set.seed(24)
    x <- runif(n) %>% as.matrix()
    z <- runif(n) %>% as.matrix()
    y <- exp(as.matrix(0.5*x + rnorm(n, 0, 0.1*z) + 1))
    data = data.frame(x = x, z = z)
    
    nr_params <- attr(make_torch_dist(dist), "nrparams_dist")
    
    list_of_formulas <- switch(nr_params,
                               "1" = list(~ 1 + x),
                               "2" = list(~ 1 + x, ~ 1 + z))
    
      mod <- deepregression(
        y = y,
        data = data,
        # define how parameters should be modeled
        list_of_formulas = list_of_formulas,
        list_of_deep_models = NULL,
        family = dist, seed = 44, engine = "torch")
    
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
