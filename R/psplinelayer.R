### instantiate spline layer

#' Wrapper function to create spline layer
#'
#' @param object or layer object.
#' @param name An optional name string for the layer (should be unique).
#' @param trainable logical, whether the layer is trainable or not.
#' @param input_shape Input dimensionality not including samples axis.
#' @param P a penalty matrix
#' @param kernel_initializer function to initialize the kernel (weight). Default
#' is "glorot_uniform".
#' @param bias_initializer function to initialize the bias. Default is 0.
#' @param variational logical, if TRUE, priors corresponding to the penalties
#' and posteriors as defined in \code{posterior_fun} are created
#' @param diffuse_scale diffuse scale prior for scalar weights
#' @param posterior_fun function defining the variational posterior
#' @param output_dim the number of units for the layer
#' @param ... further arguments passed to \code{args} used in \code{create_layer}
layer_spline <- function(object,
                         name = NULL,
                         trainable = TRUE,
                         input_shape,
                         regul = NULL,
                         P,
                         kernel_initializer = 'glorot_uniform',
                         variational = FALSE,
                         # prior_fun = NULL,
                         posterior_fun = NULL,
                         diffuse_scale = 1000,
                         output_dim = 1L,
                         ...) {

  if(variational){
    bigP = bdiag(lapply(1:length(Ps), function(j){
      # return vague prior for scalar
      if(length(Ps[[j]])==1) return(diffuse_scale^2) else
        return(invertP(Ps[[j]]))}))
  }else{
      bigP = bdiag(Ps)
  }

  if(!is.null(regul) | variational)
    regul <- NULL else
      regul <- function(x)
        k_summary(k_batch_dot(x, k_dot(
          # tf$constant(
          sparse_mat_to_tensor(bigP),
          # dtype = "float32"),
          x),
          axes=2) # 1-based
        )

    args <- c(list(input_shape = input_shape),
              name = name,
              units = output_dim,
              trainable = trainable,
              kernel_regularizer=regul,
              use_bias=use_bias,
              list(...))

    if(variational){

      class <- tfprobability::tfp$layers$DenseVariational
      args$make_posterior_fn = posterior_fun
      args$make_prior_fn = function(kernel_size,
                                    bias_size = 0L,
                                    dtype) prior_pspline(kernel_size = kernel_size,
                                                         bias_size = bias_size,
                                                         dtype = 'float32',
                                                         P = as.matrix(bigP))
      args$regul <- NULL

    }else{

      class <- tf$keras$layers$Dense
      args$kernel_initializer=kernel_initializer
      args$bias_initializer='zeros'

    }

    create_layer(layer_class = class,
                 object = object,
                 args = args
    )
}


invertP <- function(mat){
  
  invmat <- tryCatch(chol2inv(chol(mat)),
                     error = function(e) chol2inv(chol(mat + diag(rep(1, ncol(mat))) * 1e-09)))
  return(invmat)
  
}
#### get layer based on smoothCon object
get_layers_from_s <- function(this_param, nr=NULL, variational=FALSE,
                              posterior_fun=NULL, trafo=FALSE, #, prior_fun=NULL
                              output_dim = 1, k_summary = k_sum,
                              return_layer = TRUE, fsbatch_optimizer = FALSE,
                              nobs
                              )
{

  if(is.null(this_param)) return(NULL)

  lambdas <- list()
  Ps = list()
  params = 0

  # create vectors of lambdas and list of penalties
  if(!is.null(this_param$linterms)){
    lambdas = c(lambdas, as.list(rep(0, ncol_lint(this_param$linterms))))
    
    Ps = c(Ps, list(matrix(0, ncol=1, nrow=1))[rep(1, ncol_lint(this_param$linterms))])
    params = ncol(this_param$linterms)
  }
  if(!is.null(this_param$smoothterms)){
    these_lambdas = sapply(this_param$smoothterms, function(x) x[[1]]$sp)
    lambdas = c(lambdas, these_lambdas)
    these_Ps = lapply(this_param$smoothterms, function(x){ 
      
      if(length(x[[1]]$S)==1 & is.null(x[[1]]$by.level)) return(x[[1]]$S)
      if(length(x[[1]]$S)==2 & !is.null(length(x[[1]]$margin))) return(x[[1]]$S)
      return(list(Matrix::bdiag(lapply(x,function(y)y$S[[1]]))))
      
    })
    # is_TP <- sapply(these_Ps, length) > 1
    # if(any(is_TP))
    #   these_Ps[which(is_TP)] <- lapply(these_Ps[which(is_TP)],
    #                                    function(x){
    # 
    #                                      return(
    #                                       # isotropic smoothing
    #                                       # TODO: Allow f anisotropic smoothing
    #                                        x[[1]] + x[[2]]
    #                                       # kronecker(x[[1]],
    #                                       #           diag(ncol(x[[2]]))) +
    #                                       #   kronecker(diag(ncol(x[[1]])), x[[2]])
    #                                      )
    #                                    })
    # s_as_list <- sapply(these_Ps, class)=="list"
    # if(any(s_as_list)){
    #   for(i in which(s_as_list)){
    #     if(length(these_Ps[i])> 1)
    #       stop("Can not deal with penalty of smooth term ", names(these_Ps)[i])
    #     these_Ps[i] <- these_Ps[i][[1]]
    #   }
    # }
    Ps = c(Ps, these_Ps)
  }else{
    # only linear terms
    # name <- "linear"
    # if(!is.null(nr)) name <- paste(name, nr, sep="_")
    return(
      # layer_dense(input_shape = list(params),
      #             units = 1,
      #             use_bias = FALSE,
      #             name = name)
      NULL
    )
  }
  if(!is.null(this_param$linterms)){
    zeros <- ncol_lint(this_param$linterms)
    mask <- as.list(rep(0,zeros))
    mask <- c(mask, as.list(rep(1,length(lambdas)-zeros)))
  }else{
    mask <- as.list(rep(1,length(lambdas)))
  }
  
  name <- "structured_nonlinear"
  if(!is.null(nr)) name <- paste(name, nr, sep="_")

  if(trafo){

    return(bdiag(lapply(1:length(Ps), function(j) lambdas[[j]] * Ps[[j]][[1]])))

  }

  if(all(unlist(sapply(lambdas, function(x) x==0))))
    regul <- 0 else regul <- NULL
  
  if(!return_layer) return(list(Ps=Ps, lambdas = lambdas)) 
  
  if(fsbatch_optimizer){
    return(trainable_pspline(units = output_dim, 
                             this_lambdas = unname(lapply(1:length(lambdas), function(j)
                               lambdas[[j]]*mask[[j]])), 
                             this_mask = mask,
                             this_P = lapply(unname(Ps), function(x) 
                               tf$linalg$LinearOperatorFullMatrix(reduce_one_list(x))), 
                             this_n = nobs,
                             this_nr = nr))
  }
  
  # put together lambdas and Ps
  Ps <- lapply(1:length(Ps), function(j){ 
    if(length(lambdas[[j]])==1){ 
      
      if(is.list(Ps[[j]]))
        return(lambdas[[j]] * Ps[[j]][[1]]) else return(lambdas[[j]] * Ps[[j]])
    }
    Reduce("+", lapply(1:length(lambdas[[j]]), function(k) lambdas[[j]][k] * Ps[[j]][[k]]))
  })
  
  params = params + sum(sapply(these_Ps, NCOL))
  
  return(
    layer_spline(input_shape = list(as.integer(params)),
                 # the one is just an artifact from concise
                 name = name,
                 regul = regul,
                 Ps = Ps,
                 variational = variational,
                 posterior_fun = posterior_fun,
                 output_dim = output_dim,
                 k_summary = k_summary)
  )

}

combine_lambda_and_penmat <- function(lambdas, Ps)
{
  
  bigP <- list()
  
  for(i in seq_along(length(lambdas)))
  {
    
    if(is.list(lambdas)){
      bigP[[i]] <- do.call("+", combine_lambda_and_penmat(lambdas[[i]], Ps[[i]]))
    }else{
      bigP[[i]] <- lambdas[[i]]*Ps[[i]]
    }
    
  }
  
  return(bigP)
  
}

tf_block_diag <- function(listMats)
{
  lob = lapply(listMats, function(x) tf$linalg$LinearOperatorFullMatrix(x))
  res = tf$linalg$LinearOperatorBlockDiag(lob)
  return(res)
}



gam_plot_data <- function(pp, weights, grid_length = 40)
{
  
  org_values <- pp$get_org_values()
  
  if(length(org_values)==1){
    
    BX <- pp$data_trafo()
    
    plotData <-
      list(org_feature_name = pp$term,
           value = org_values[[1]],
           design_mat = BX,
           coef = weights,
           partial_effect = BX%*%weights)
    
  }else{
    
    BX <- pp$data_trafo()
    
    plotData <-
      list(org_feature_name = pp$term,
           value = do.call("cbind", org_values),
           design_mat = BX,
           coef = weights
           )
    
    this_x <- do.call(seq, c(as.list(range(plotData$value[,1])),
                             list(l=grid_length)))
    this_y <- do.call(seq, c(as.list(range(plotData$value[,2])),
                             list(l=grid_length)))
    df <- as.data.frame(expand.grid(this_x, this_y))
    colnames(df) <- extractvar(pp$term)
    pmat <- pp$predict_trafo(newdata = df)
    plotData$df <- df
    plotData$x <- this_x
    plotData$y <- this_y
    plotData$partial_effect <- pmat%*%weights
    
  }
  
  return(plotData)
  
}

# CustomLayer <- R6::R6Class("penalizedLikelihoodLoss",
#                            
#                            inherit = KerasLayer,
#                            
#                            public = list(
#                              
#                              self$lambdas = NULL
#                              self$penalty = NULL
#                              
#                              initialize = function(lambdas, Ps, model) {
#                                self$lambdas <- lapply(lambdas, function(l) 
#                                  if(is.list(l)) lapply(l, function(ll) tf$Variable(ll)) else
#                                    tf$Variable(ll))
#                                bigP <- tf_block_diag(combine_lambda_and_penmat(self$lambdas, Ps))
#                                self$penalty <- function(x) k_sum(k_batch_dot(x, k_dot(
#                                  # tf$constant(
#                                  bigP,
#                                  # dtype = "float32"),
#                                  x),
#                                  axes=2) # 1-based
#                                )
#                              },
#                              
#                              get_lambdas = function() {
#                                return(self$lambdas)
#                              },
#                              
#                              get_penalty = function {
#                                return(self$penalty)
#                              },
#                              
#                              custom_loss = function(y, model) {
#                                (model %>% tfd_log_prob(y)) + self$penalty() 
#                              },
#                              
#                              call = function(x, mask = NULL) {
#                                k_dot(x, self$kernel)
#                              },
#                              
#                              compute_output_shape = function(input_shape) {
#                                list(input_shape[[1]], self$output_dim)
#                              }
#                            )
# )

layer_spline = function(units = 1L, P, name) {
  python_path <- system.file("python", package = "deepregression")
  splines <- reticulate::import_from_path("psplines", path = python_path)
  
  return(splines$layer_spline(P = as.matrix(P), units = units, name = name))
}

trainable_pspline = function(units, this_lambdas, this_mask, this_P, this_n, this_nr) {
  python_path <- system.file("python", package = "deepregression")
  splines <- reticulate::import_from_path("psplines", path = python_path)
  
  return(splines$PenLinear(units, lambdas = this_lambdas, mask = this_mask,
                           P = this_P, n = this_n, nr = this_nr))
}

# kerasGAM = function(inputs, outputs) {
#   python_path <- system.file("python", package = "deepregression")
#   splines <- reticulate::import_from_path("psplines", path = python_path)
#   
#   return(splines$kerasGAM(inputs, outputs))
# }

build_kerasGAM = function(factor, lr_scheduler, avg_over_past, constantdiv = 0, constantinv = 0, constinv_scheduler = NULL) {
  python_path <- system.file("python", package = "deepregression")
  splines <- reticulate::import_from_path("psplines", path = python_path)
  
  return(splines$build_kerasGAM(
    # Plist = Plist,
    fac = factor, 
    lr_scheduler = lr_scheduler, 
    avg_over_past = avg_over_past))
}


tf_incross = function(w, P) {
  python_path <- system.file("python", package = "deepregression")
  splines <- reticulate::import_from_path("psplines", path = python_path)
  
  return(splines$tf_incross(w, P))
}

tp_penalty <- function(P1,P2,lambda1,lambda2=NULL)
{
  
  if(is.null(lambda2)) lambda2 <- lambda1
  return(lambda1 * kronecker(P1, diag(ncol(P2))) + lambda2 * kronecker(diag(ncol(P2)), P1))
  
}

quadpen <- function(P){
  retfun <- function(w) tf$reduce_sum(tf$matmul(
    tf$matmul(tf$transpose(w), tf$cast(P, "float32")), w))
  return(retfun)
}