prior_trainable <-
  function(kernel_size,
           bias_size = 0L,
           dtype = NULL,
           diffuse_scale = 1000) {
    n <- kernel_size + bias_size
    keras_model_sequential() %>%
      layer_variable(n, dtype = dtype, trainable = TRUE) %>%
      layer_distribution_lambda(function(t) {
        tfd_independent(tfd_normal(loc = t, scale = diffuse_scale),
                        reinterpreted_batch_ndims = 1L)
      })
  }


posterior_mean_field <-
  function(kernel_size,
           bias_size = 0L,
           dtype = NULL) {
    n <- kernel_size + bias_size
    keras_model_sequential(list(
      layer_variable(shape = as.integer(2 * n), dtype = dtype),
      layer_distribution_lambda(
        make_distribution_fn = function(t) {
          tfd_independent(tfd_normal(
            loc = t[1:n],
            scale = 1e-8 + tf$nn$softplus(log(expm1(1)) + t[(n + 1):(2 * n)])
          ), reinterpreted_batch_ndims = 1L)
        }
      )
    ))
  }

prior_pspline <- 
  function(kernel_size,
           dtype = 'float32',
           P) {
    n <- as.integer(kernel_size)
    keras_model_sequential() %>% 
      layer_variable(n, dtype = dtype, trainable = TRUE) %>% 
      layer_distribution_lambda(function(t) {
        tfd_multivariate_normal_full_covariance(
          loc = tf$zeros(shape = c(n, 1L)), 
          covariance_matrix = tf$constant(P, dtype="float32")
          )
      })
    
  }

prior_dense <- function(kernel_size, 
                                   bias_size = 0L,
                                   diffuse_scale = diffuse_scale^2)
{
  
  n <- as.integer(kernel_size + bias_size)
  keras_model_sequential() %>% 
    layer_variable(n, trainable = TRUE) %>% 
    layer_distribution_lambda(function(t) {
      tfd_independent(
        tfd_normal(
          location = tf$zeros(shape = c(n, 1L)),
          scale = tf$constant(diffuse_scale)
        )
      )
    })
  
}