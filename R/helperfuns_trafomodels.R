eval_bsp <- function(y, order = 3, supp = range(y)) {
  
  # Evaluate a Bernstein Polynom (bsp) with a predefined order on a predefined 
  # support that is used for scaling since a Beta distribution is defined on (0,1).
  # MLT Vignette p. 9
  #
  # y: numeric vector of length n
  # order: postive integer which is called M in the literature
  # supp: numeric vector of length 2 that is used to scale y down to (0,1)
  #
  # returns a numeric matrix (n x (order + 1))
  
  y <- (y - supp[1]) / diff(supp)
  sapply(0:order, function(m) dbeta(y, m + 1, order + 1 - m) / (order + 1))
  
}

eval_bsp_prime <- function(y, order = 3, supp = range(y)) {
  
  # Evaluate the first derivative of the bsp with a predefined order on a predefined 
  # support that is used for scaling since a Beta distribution is defined on (0,1).
  # MLT Vignette p. 9. Note that "order" cancels out and that (1/diff(y_var$support)^deriv)
  # is multiplied afterwards. This is only due to numerical reasons to get the 
  # exact same quantities as mlt::mlt. Furthermore, order/(order - 1 + 1) cancels
  # out in the mlt::mlt implementation which is not as stated on p. 9.
  #
  # y: numeric vector of length n
  # order: postive integer which is called M in the literature
  # supp: numeric vector of length 2 that is used to scale y down to (0,1)
  #
  # returns a numeric matrix (n x (order + 1))
  
  y <- (y - supp[1]) / diff(supp)
  sapply(0:order, function(m) {
    
    first_t <- dbeta(y, m, order - m + 1) / order
    sec_t <- dbeta(y, m + 1, order - m) / order
    
    first_t[is.infinite(first_t)] <- 0L
    sec_t[is.infinite(sec_t)] <- 0L
    
    (first_t - sec_t) * order
  })
}

# TensorFlow repeat function which is not available for TF 2.0
tf_repeat <- function(a, dim)
  tf$reshape(tf$tile(tf$expand_dims(a, axis = -1L),  c(1L, 1L, dim)), shape = list(-1L, ncol(a)[[1]]*dim))

# Row-wise tensor product using TensorFlow
tf_row_tensor <- function(a,b)
{
  tf$multiply(
    tf_row_tensor_left_part(a,b),
    tf_row_tensor_right_part(a,b)
  )
}

tf_row_tensor_left_part <- function(a,b)
{
  tf_repeat(a, ncol(b)[[1]])
}

tf_row_tensor_right_part <- function(a,b)
{
  tf$tile(b, c(1L, ncol(a)[[1]]))
}

###############################################################################################
# for trafo with interacting features

mono_trafo_multi <- function(w, bsp_dim) 
{
  
  w_res <- tf$reshape(w, shape = list(bsp_dim, as.integer(nrow(w)/bsp_dim)))
  w1 <- tf$slice(w_res, c(0L,0L), size=c(1L,ncol(w_res)))
  wrest <- tf$math$softplus(tf$slice(w_res, c(1L,0L), size=c(as.integer(nrow(w_res)-1),ncol(w_res))))
  w_w_cons <- tf$cumsum(k_concatenate(list(w1,wrest), 
                                      axis = 1L # this is 1 and not 0 because k_concat is 1-based
                                      ), axis=0L)
  return(tf$reshape(w_w_cons, shape = list(nrow(w),1L)))
  
}

MonoMultiLayer <- R6::R6Class("MonoMultiLayer",
                              
                              inherit = KerasLayer,
                              
                              public = list(
                                
                                output_dim = NULL,
                                
                                kernel = NULL,
                                
                                dim_bsp = NULL,
                                
                                initialize = function(output_dim, dim_bsp) {
                                  self$output_dim <- output_dim
                                  self$dim_bsp <- dim_bsp
                                },
                                
                                build = function(input_shape) {
                                  self$kernel <- self$add_weight(
                                    name = 'kernel', 
                                    shape = list(input_shape[[2]], self$output_dim),
                                    initializer = initializer_random_normal(),
                                    trainable = TRUE
                                  )
                                },
                                
                                call = function(x, mask = NULL) {
                                  tf$matmul(x, mono_trafo_multi(self$kernel, self$dim_bsp))
                                },
                                
                                compute_output_shape = function(input_shape) {
                                  list(input_shape[[1]], self$output_dim)
                                }
                              )
)

# define layer wrapper function
layer_mono_multi <- function(object, 
                             input_shape = NULL,
                             output_dim = 1L,
                             dim_bsp = NULL,
                             name = "constraint_mono_layer_multi", 
                             trainable = TRUE
) {
  create_layer(MonoMultiLayer, object, list(
    name = name,
    trainable = trainable,
    input_shape = input_shape,
    output_dim = as.integer(output_dim),
    dim_bsp = as.integer(dim_bsp)
  ))
}


########## other version

MonoMultiTrafoLayer <- R6::R6Class("MonoMultiTrafoLayer",
                              
                              inherit = KerasLayer,
                              
                              public = list(
                                
                                output_dim = NULL,
                                
                                kernel = NULL,
                                
                                dim_bsp = NULL,
                                
                                initialize = function(output_dim, dim_bsp) {
                                  self$output_dim <- output_dim
                                  self$dim_bsp <- dim_bsp
                                },
                                
                                build = function(input_shape) {
                                  self$kernel <- self$add_weight(
                                    name = 'kernel', 
                                    shape = list(input_shape[[2]], self$output_dim),
                                    initializer = initializer_random_normal(),
                                    trainable = TRUE
                                  )
                                },
                                
                                call = function(x, mask = NULL) {
                                  tf$multiply(x, tf$transpose(mono_trafo_multi(self$kernel, self$dim_bsp)))
                                },
                                
                                compute_output_shape = function(input_shape) {
                                  list(input_shape[[1]], self$output_dim)
                                }
                              )
)

# define layer wrapper function
layer_mono_multi_trafo <- function(object, 
                                   input_shape = NULL,
                                   output_dim = 1L,
                                   dim_bsp = NULL,
                                   name = "constraint_mono_layer_multi_trafo", 
                                   trainable = TRUE
) {
  create_layer(MonoMultiTrafoLayer, object, list(
    name = name,
    trainable = trainable,
    input_shape = input_shape,
    output_dim = as.integer(output_dim),
    dim_bsp = as.integer(dim_bsp)
  ))
}

# to retrieve the weights on their original scale
softplus <- function(x) log(exp(x)+1)
reshape_softplus_cumsum <- function(x, order_bsp_p1)
{
  
  x <- matrix(x, nrow = order_bsp_p1, byrow=T)
  x[2:nrow(x),] <- softplus(x[2:nrow(x),])
  apply(x, 2, cumsum)
  
}

correct_min_val <- function(pcf, addconst = 10)
{

  minval <- suppressWarnings(min(pcf$linterms[,sapply(pcf$linterms, is.numeric)], na.rm = T))
  if(!is.null(pcf$smoothterms))
    minval <- min(c(minval, 
                    suppressWarnings(sapply(pcf$smoothterms,
                                            function(x) min(x[[1]]$X)))))
  if(minval<0)
  {
    
    minval <- minval - addconst
    
    if(!is.null(pcf$linterms))
      pcf$linterms[,sapply(pcf$linterms, is.numeric)] <- 
        pcf$linterms[,sapply(pcf$linterms, is.numeric)] - minval
    if(!is.null(pcf$smoothterms))
      pcf$smoothterms <- lapply(pcf$smoothterms, function(x){
        x[[1]]$X <- x[[1]]$X - minval
        return(x)
      })
    
    
  }else{
    
    return(pcf)
    
  }
  
  if(minval==Inf) return(pcf)
  
  attr(pcf,"minval") <- minval
    
  return(pcf)
  
}

secondOrderPenBSP <- function(order_bsp, order_diff = 2)
{
  
  # taken from https://github.com/cran/penDvine/blob/master/R/pen.matrix.r
  
  if(order_diff == 0){
    
    k.dim <- order_bsp + 1
    k <- k.dim-1
    
    c2 <- factorial(k+1)/factorial(k-2)
    A <- matrix(0,k.dim-2,k.dim)
    diag(A) <- 1
    diag(A[,-1]) <- -2
    diag(A[,-c(1,2)]) <- 1
    
    A.hat <- matrix(NA,k.dim-2,k.dim-2)
    for(i in 0:(k-2)) {
      i.z <- i+1
      for(j in 0:(k-2)) {
        j.z <- j+1
        A.hat[i.z,j.z] <- choose(k-2,j)*choose(k-2,i)*beta(i+j+1,2*k-i-j-3)
      }
    }  
    
    return(c2^2*(t(A)%*%A.hat%*%A))
    
  }
  
  K <- order_bsp+1

  if(order_diff==1) {
    L <- diag(1,K)
    L.1 <- diag(-1,K,K-1)
    L.2 <- matrix(0,K,1)
    L1 <- cbind(L.2,L.1)
    L <- L+L1
    L <- L[1:(K-1),]
  }
  if(order_diff==2) {
    L <- diag(1,K,K)
    L1 <- diag(-2,K,(K-1))
    L2 <- diag(1,K,(K-2))
    L.1 <- matrix(0,K,1)
    L1 <- cbind(L.1,L1)
    L2 <- cbind(L.1,L.1,L2)
    L <- L+L1+L2
    L <- L[1:(K-2),]
  }
  if(order_diff==3) {
    L <- diag(1,(K-3),(K-3))
    L.help <- matrix(0,(K-3),1)
    L1 <- diag(-3,(K-3),(K-3))
    M1 <- cbind(L,L.help,L.help,L.help)
    M2 <- cbind(L.help,L1,L.help,L.help)
    M3 <- cbind(L.help,L.help,-L1,L.help)
    M4 <- cbind(L.help,L.help,L.help,-L)
    L <- (M1+M2+M3+M4)
  }
  if(order_diff==4) {
    L <- diag(1,(K-4),(K-4))
    L.help <- matrix(0,(K-4),1)
    L1 <- diag(-4,(K-4),(K-4))
    L2 <- diag(6,(K-4),(K-4))
    M1 <- cbind(L,L.help,L.help,L.help,L.help)
    M2 <- cbind(L.help,L1,L.help,L.help,L.help)
    M3 <- cbind(L.help,L.help,L2,L.help,L.help)
    M4 <- cbind(L.help,L.help,L.help,L1,L.help)
    M5 <- cbind(L.help,L.help,L.help,L.help,L)
    L <- (M1+M2+M3+M4+M5)
  }
  
  return(crossprod(L))
  
}

calculate_log_score <- function(x, output)
{
  
  if(is.character(x$init_params$base_distribution) &
     x$init_params$base_distribution=="normal"){
    bd <- tfd_normal(loc = 0, scale = 1)
  }else if((is.character(x$init_params$base_distribution) & 
            x$init_params$base_distribution=="logistic")){ 
    bd <- tfd_logistic(loc = 0, scale = 1)
  }else{
    bd <- x$init_params$base_distribution
  }
  return(
    as.matrix(bd %>% tfd_log_prob(output[,2,drop=F] + output[,1,drop=F])) +  
      as.matrix(log(tf$clip_by_value(output[,3,drop=F], 1e-8, Inf)))
  )
  
}
