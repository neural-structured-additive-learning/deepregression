context("deepregression layers Torch")

test_that("lasso layers", {
  
  set.seed(42)
  torch_manual_seed(42)
  n <- 1000
  p <- 50
  x <- runif(n*p) %>% matrix(ncol=p)
  y <- 10*x[,1] + rnorm(n)
  
  input <- torch_tensor(x)
  data_x <- input
  data_y <- torch_tensor(y)
  
  tib_module <- tib_layer_torch(units = 1, la = 1, input_shape = 50)
  optimizer_tiblasso <- optim_adam(tib_module$parameters, lr=1e-2)
  
  batch_size <- 32
  num_data_points <- data_y$size(1)
  num_batches <- floor(num_data_points/batch_size)
  
  epochs <- 250
  for(epoch in 1:epochs){
    l <- c()
    # rearrange the data each epoch
    permute <- torch_randperm(num_data_points) + 1L
    data_x <- data_x[permute]
    data_y <- data_y[permute]
    # manually loop through the batches
    for(batch_idx in 1:num_batches){
      # here index is a vector of the indices in the batch
      index <- (batch_size*(batch_idx-1) + 1):(batch_idx*batch_size)
      
      x_batch <- data_x[index]
      y_batch <- data_y[index]
      
      optimizer_tiblasso$zero_grad()
      output <- tib_module(x_batch) %>% torch_flatten()
      l_tib <- nnf_mse_loss(output, y_batch)
      l_tib$backward()
      optimizer_tiblasso$step()
      l <- c(l, l_tib$item())
    }
    }

  expect_true(
    abs(Reduce(x = apply(
      cbind(t(as.array(tib_module$parameters[[1]])),
            as.array(tib_module$parameters[[2]])),
      MARGIN = 1, 
      prod),
      f = "prod")) < 1)
  
})
