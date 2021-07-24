#' @title Fitting Semi-Structured Transformation Models
#'
#' @param y response variable
#' @param data data.frame or named list with input features
#' @param list_of_formulas a list of formulas for the shift term (h_2),
#' the interaction term (h_1) and, if, specified a third entry to specify 
#' lags for SATs models. See \code{?deepregression} for more details how
#' formulas can be specified.
#' @param list_of_deep_models a named list of functions
#' specifying a keras model.
#' @param order_bsp NULL or integer; order of Bernstein polynomials; if not NULL,
#' a conditional transformation model (CTM) is fitted.
#' @param y_basis_fun,y_basis_fun_prime basis functions for y transformation for CTM case
#' @param split_between_shift_and_theta if \code{family == 'transformation_model'} and
#' \code{!is.null(train_together)}, \code{split_between_shift_and_theta} is supposed
#' to define how many of the last layer's hidden units are used for the shift
#' term and how many for the theta term (an integer vector of length 2).

#' @param addconst_interaction positive constant;
#' a constant added to the additive predictor of the interaction term.
#' If \code{NULL}, terms are left unchanged. If 0 and predictors have negative values in their
#' design matrix, the minimum value of all predictors is added to ensure positivity.
#' If > 0, the minimum value plus the \code{addconst_interaction} is added to each predictor
#' in the interaction term.
#' @param base_distribution tfd_distribution or string; the base distribution for 
#' transformation models, see \code{?deeptransformation\_init}.
#' @param ... further arguments passed to the \code{deepregression\_init} function
#'
#' @import tensorflow tfprobability keras mgcv dplyr R6 reticulate Matrix
#'
#' @importFrom keras fit compile
#' @importFrom Metrics auc
#' @importFrom tfruns is_run_active view_run_metrics update_run_metrics write_run_metadata
#' @importFrom graphics abline filled.contour matplot par points
#' @importFrom stats as.formula model.matrix terms terms.formula uniroot var dbeta coef predict
#' @importFrom methods slotNames is as
#'
#' @export
deeptransformation <- function(
  y, data,
  list_of_formulas,
  list_of_deep_models,
  order_bsp = NULL,
  y_basis_fun = function(y) eval_bsp(y, order = order_bsp, supp = range(y)),
  y_basis_fun_prime = function(y) eval_bsp_prime(y, order = order_bsp,
                                                 supp = range(y)) / diff(range(y)),
  split_between_shift_and_theta = NULL,
  addconst_interaction = NULL,
  base_distribution = "normal",
  ...
)
{
  
  if(family=="transformation_model" & !is.null(addconst_interaction)){
    # ensure positivity of interaction
    parsed_formulas_contents[[2]] <- correct_min_val(parsed_formulas_contents[[2]],
                                                     addconst_interaction)
    if(is.null(attr(parsed_formulas_contents[[2]], "minval")))
      addconst_interaction <- NULL else
        addconst_interaction <- attr(parsed_formulas_contents[[2]], "minval")
      # if(minval<0) y <- y - minval
  }
  
  # atm_toplayer <- NULL
  
  if(nr_params==3){
    
    pos_lags <- sum(sapply(parsed_formulas_contents, function(x) (
      !is.null(x$deepterms))*1 + !is.null(x$linterms)))
    intercept_form <- "(Intercept)" %in% names(parsed_formulas_contents[[3]]$linterms)
    atm_lags <- ncol(input_cov[[pos_lags]])-intercept_form
    
    input_cov[[pos_lags]] <- lapply(1:atm_lags, 
                                    function(i)y_basis_fun(input_cov[[pos_lags]][,i+intercept_form]))
    input_cov <- unlist_order_preserving(input_cov)
    if(!is.null(validation_data)){
      validation_data[[1]][[3]] <-
        lapply(1:(ncol(validation_data[[1]][[pos_lags]])-intercept_form), 
               function(i)y_basis_fun(validation_data[[1]][[pos_lags]][,i+intercept_form]))
      validation_data[[1]] <- unlist_order_preserving(validation_data[[1]])
    }
    
    ncol_structured <- ncol_structured[1:2]
    
    # atm_toplayer <- list_structured[[3]]
    # list_structured <- list_structured[1:2]
    
  }
  
  input_cov <- c(input_cov,
                 list(y_basis_fun(y),
                      y_basis_fun_prime(y)))
  
  if(!is.null(validation_data))
    validation_data <- list(unname(
      c(validation_data[[1]], 
        list(y_basis_fun(validation_data[[2]]),
             y_basis_fun_prime(validation_data[[2]])))
    ), validation_data[[2]])
  
  model <- deeptransformation_init(
    n_obs = n_obs,
    ncol_structured = ncol_structured,
    ncol_deep = ncol_deep,
    list_structured = list_structured,
    list_deep = list_of_deep_models,
    orthogX = orthogX,
    lambda_lasso = lambda_lasso,
    lambda_ridge = lambda_ridge,
    monitor_metric = monitor_metric,
    optimizer = optimizer,
    split_fun = split_fun,
    orthog_fun = orthog_fun,
    order_bsp = order_bsp,
    train_together = train_together_ind(train_together),
    split_between_shift_and_theta = split_between_shift_and_theta,
    addconst_interaction = addconst_interaction,
    base_distribution = base_distribution,
    atm_lags = ifelse(nr_params==2,0,atm_lags),
    ...
  )
  
  ret$init_params <- c(ret$init_params,
                       order_bsp = order_bsp,
                       y_basis_fun = y_basis_fun,
                       y_basis_fun_prime = y_basis_fun_prime,
                       base_distribution = base_distribution
  )
  
  class(ret) <- c("deeptrafo", class(ret))
  
}



#' @title Initializing Deep Transformation Models
#'
#'
#' @param n_obs number of observations
#' @param ncol_structured a vector of length #parameters
#' defining the number of variables used for each of the parameters.
#' If any of the parameters is not modelled using a structured part
#' the corresponding entry must be zero.
#' @param ncol_deep a vector of length #parameters
#' defining the number of variables used for each of the parameters.
#' If any of the parameters is not modelled using a deep part
#' the corresponding entry must be zero. If all parameters
#' are estimated by the same deep model, the first entry must be
#' non-zero while the others must be zero.
#' @param list_structured list of (non-linear) structured layers
#' (list length between 0 and number of parameters)
#' @param list_deep list of deep models to be used
#' (list length between 0 and number of parameters)
#' @param lambda_lasso penalty parameter for l1 penalty of structured layers
#' @param lambda_ridge penalty parameter for l2 penalty of structured layers
#' @param weights observation weights used in the likelihood
#' @param learning_rate learning rate for optimizer
#' @param optimizer optimizer used (defaults to adam)
#' @param monitor_metric see \code{?deepregression}
#' @param orthog_fun function defining the orthogonalization
#' @param orthogX vector of columns defining the orthgonalization layer
#' @param split_fun see \code{?deepregression}
#' @param order_bsp NULL or integer; order of Bernstein polynomials; if not NULL,
#' a conditional transformation model (CTM) is fitted.
#' @param use_bias_in_structured whether or not to use a bias in structured
#' layers
#' @param train_together see \code{?deepregression}
#' @param split_between_shift_and_theta see \code{?deepregression}
#' @param interact_pred_trafo specifies a transformation function applied
#' to the interaction predictor using a layer lambda (e.g. to ensure positivity)
#' @param addconst_interaction additive constant added to predictor matrix to
#' ensure positivity
#' @param penalize_bsp scalar value > 0; amount of penalization of Bernstein polynomials
#' @param order_bsp_penalty integer; order of Bernstein polynomial penalty. 0 results in a
#' penalty based on integrated squared second order derivatives, values >= 1 in difference
#' penalties
#' @param base_distribution a string ("normal", "logistic") or TFP distribution; 
#' the base distribution for the transformation model. 
#' Per default \code{tfd_normal(loc = 0, scale = 1)} but any other distribution is possible (e.g.,
#' \code{tfd_logistic(loc = 0, scale = 1)}).
#'
#' @export
#'
deeptransformation_init <- function(
  n_obs,
  ncol_structured,
  ncol_deep,
  list_structured,
  list_deep,
  lambda_lasso=NULL,
  lambda_ridge=NULL,
  weights = NULL,
  learning_rate = 0.01,
  optimizer = optimizer_adam(lr = learning_rate),
  monitor_metric = list(),
  orthog_fun = orthog,
  orthogX = NULL,
  split_fun = split_model,
  order_bsp,
  use_bias_in_structured = FALSE,
  train_together = NULL,
  split_between_shift_and_theta = NULL,
  interact_pred_trafo = NULL,
  addconst_interaction = NULL,
  penalize_bsp = 0,
  order_bsp_penalty = 2,
  base_distribution = "normal",
  batch_shape = NULL,
  atm_lags = 0,
  atm_toplayer = NULL
)
{
  
  nr_params = 2 # shift & interaction term
  output_dim = rep(1, nr_params) # only univariate responses
  # if(length(list_deep)==1 & is.null(list_deep[[1]]))
  #   list_deep <- list_deep[rep(1,2)]
  
  
  # define the input layers
  inputs_deep <- lapply(ncol_deep, function(param_list){
    if(is.list(param_list) & length(param_list)==0) return(NULL)
    lapply(param_list, function(nc){
      
      if(!is.null(batch_shape)) shape <- NULL else{
        
        if(sum(unlist(nc))!=0){
          if(is.list(nc) & length(nc)>1)
            shape <- list(as.integer(sum(unlist(nc)))) else if(is.list(nc) & length(nc)==1)
              shape <- as.list(as.integer(nc[[1]]))
        }else{
          return(NULL)
        }     
      }
      return(
        layer_input(shape = shape,
                    batch_shape = batch_shape)
      )
    })
  })
  
  inputs_struct <- lapply(1:length(ncol_structured), function(i){
    nc = ncol_structured[i]
    if(nc==0) return(NULL) else
      # if(!is.null(list_structured[[i]]) & nc > 1)
      # nc>1 will cause problems when implementing ridge/lasso
      layer_input(shape = list(as.integer(nc)))
  })
  
  if(!is.null(orthogX)){
    ox <- lapply(1:length(orthogX), function(i){
      
      x = orthogX[[i]]
      if(is.null(x) | is.null(inputs_deep[[i]])) return(NULL) else{
        lapply(x, function(xx){
          if(is.null(xx) || xx==0) return(NULL) else
            return(layer_input(shape = list(as.integer(xx))))})
      }
    })
  }
  
  # inputs for BSP trafo of Y, both n x tilde{M}
  input_theta_y <- layer_input(shape = list(order_bsp+1L))
  input_theta_y_prime <- layer_input(shape = list(order_bsp+1L))
  if(atm_lags) input_theta_atm <- lapply(1:atm_lags, function(x) 
    layer_input(shape = list(as.integer((order_bsp+1L))))) else
      input_theta_atm <- NULL
  
  structured_parts <- vector("list", 2)
  
  # define structured predictor
  if(is.null(inputs_struct[[1]]))
  {
    structured_parts[[1]] <-  NULL
    
  }else{
    
    if(is.null(list_structured[[1]]))
    {
      if(!is.null(lambda_lasso) & is.null(lambda_ridge)){
        
        l1 = tf$keras$regularizers$l1(l=lambda_lasso)
        
        structured_parts[[1]] <- inputs_struct[[1]] %>%
          layer_dense(
            units = as.integer(output_dim[1]),
            activation = "linear",
            use_bias = use_bias_in_structured,
            kernel_regularizer = l1,
            name = paste0("structured_lasso_",
                          1))
        
      }else if(!is.null(lambda_ridge) & is.null(lambda_lasso)){
        
        l2 = tf$keras$regularizers$l2(l=lambda_ridge)
        
        structured_parts[[1]] <- inputs_struct[[1]] %>%
          layer_dense(
            units = as.integer(output_dim[1]),
            activation = "linear",
            use_bias = use_bias_in_structured,
            kernel_regularizer = l2,
            name = paste0("structured_ridge_",
                          1))
        
        
      }else if(!is.null(lambda_ridge) & !is.null(lambda_lasso)){
        
        l12 = tf$keras$regularizers$l1_l2(l1=lambda_lasso,
                                          l2=lambda_ridge)
        
        structured_parts[[1]] <- inputs_struct[[1]] %>%
          layer_dense(
            units = as.integer(output_dim[1]),
            activation = "linear",
            use_bias = use_bias_in_structured,
            kernel_regularizer = l12,
            name = paste0("structured_elastnet_",
                          1))
        
      }else{
        
        structured_parts[[1]] <- inputs_struct[[1]] %>%
          layer_dense(
            units = as.integer(output_dim[1]),
            activation = "linear",
            use_bias = use_bias_in_structured,
            name = paste0("structured_linear_",
                          1))
        
      }
      
    }else{
      
      this_layer <- list_structured[[1]]
      structured_parts[[1]] <- inputs_struct[[1]] %>% this_layer
      
    }
  }
  
  
  if(!is.null(train_together) && !is.null(list_deep) &
     !(length(list_deep)==1 & is.null(list_deep[[1]])))
    list_deep_shared <- list_deep[sapply(names(list_deep),function(nnn)
      !nnn%in%names(ncol_deep[1:nr_params]))] else
        list_deep_shared <- NULL
  
  list_deep <- lapply(ncol_deep[1:nr_params], function(param_list){
    lapply(names(param_list), function(nn){
      if(is.null(nn)) return(NULL) else
        list_deep[[nn]]
    })
  })
  
  # define deep predictor
  deep_parts <- lapply(1:length(list_deep), function(i)
    if(is.null(inputs_deep[[i]]) | length(inputs_deep[[i]])==0)
      return(NULL) else
        lapply(1:length(list_deep[[i]]), function(j)
          list_deep[[i]][[j]](inputs_deep[[i]][[j]])))
  
  if(!is.null(train_together) && !is.null(list_deep_shared) &
     any(!sapply(inputs_deep, is.null))){
    
    shared_parts <- lapply(unique(unlist(train_together)), function(i)
      list_deep_shared[[i]](
        inputs_deep[[nr_params + i]][[1]]
      ))
    
    deep_parts[[1]] <- lapply(shared_parts, function(spa) spa[
      ,1:as.integer(split_between_shift_and_theta[1]),drop=F])
    deep_parts[[2]] <- lapply(shared_parts, function(spa) spa[
      ,(as.integer(split_between_shift_and_theta[1])+1L):
        (as.integer(sum(split_between_shift_and_theta))),drop=F])
    
  }
  
  ############################################################
  ################# Apply Orthogonalization ##################
  
  # create final linear predictor per distribution parameter
  # -> depending on the presence of a deep or structured part
  # the corresponding part is returned. If both are present
  # the deep part is projected into the orthogonal space of the
  # structured part
  
  deep_top_shift <- NULL
  if(!is.null(deep_parts[[1]]))
    deep_top_shift <- list(function(x) layer_dense(x, units = 1,
                                                   activation = "linear"))[
                                                     rep(1,length(deep_parts[[1]]))]
  
  ## shift term
  final_eta_pred <- combine_model_parts(deep = deep_parts[[1]],
                                        deep_top = deep_top_shift,
                                        struct = structured_parts[[1]],
                                        ox = ox[[1]],
                                        orthog_fun = orthog_fun,
                                        shared = NULL)
  
  ## interaction term
  if(is.null(deep_parts[[2]])){
    
    deep_part_ia <- NULL
    
  }else if(is.null(ox[[2]]) | (is.null(ox[[2]][[1]]) & length(ox[[2]])==1)){
    
    if(length(deep_parts[[2]])==1)
      deep_part_ia <- deep_parts[[2]][[1]] else
        deep_part_ia <- layer_add(deep_parts[[2]])
      
  }else{
    
    deep_part_ia <- orthog_fun(deep_parts[[2]], ox[[2]])
    
  }
  
  if(is.null(deep_parts[[2]])){
    
    interact_pred <- inputs_struct[[2]]
    
  }else if(is.null(inputs_struct[[2]])){
    
    interact_pred <- deep_part_ia
    
  }else{
    
    interact_pred <- layer_concatenate(list(inputs_struct[[2]],deep_part_ia))
    
  }
  
  if(!is.null(interact_pred_trafo)){
    
    # define Gamma weights
    thetas_layer <- layer_mono_multi_trafo(input_shape =
                                             list(NULL, (order_bsp+1L)*
                                                    (ncol(interact_pred)[[1]])),
                                           dim_bsp = c(order_bsp+1L))
    
    rho_part <- tf_row_tensor_right_part(input_theta_y, interact_pred) %>%
      thetas_layer() %>%
      layer_lambda(f = interact_pred_trafo)
    
    # rho_part <- tf$add(
    #   tf$constant(matrix(
    #     c(rep(neg_shift_bsp, (ncol(interact_pred)[[1]])),
    #       rep(0, (ncol(interact_pred)[[1]])*order_bsp)
    #     ), nrow=1), dtype="float32"),
    #   rho_part)
    
    aTtheta <- tf$matmul(
      tf$multiply(tf_row_tensor_left_part(input_theta_y,
                                          interact_pred),
                  rho_part),
      tf$ones(shape = c((order_bsp+1L)*(ncol(interact_pred)[[1]]),1))
    )
    aPrimeTtheta <- tf$matmul(
      tf$multiply(tf_row_tensor_left_part(input_theta_y_prime,
                                          interact_pred),
                  rho_part),
      tf$ones(shape = c((order_bsp+1L)*(ncol(interact_pred)[[1]]),1))
    )
    
  }else{
    
    # define Gamma weights
    thetas_layer <- layer_mono_multi(input_shape =
                                       list(NULL, (order_bsp+1L)*
                                              (ncol(interact_pred)[[1]])),
                                     dim_bsp = c(order_bsp+1L))
    
    ## thetas
    AoB <- tf_row_tensor(input_theta_y, interact_pred)
    AprimeoB <- tf_row_tensor(input_theta_y_prime, interact_pred)
    
    aTtheta <- AoB %>% thetas_layer()
    aPrimeTtheta <- AprimeoB %>% thetas_layer()
    
    # if(!is.null(addconst_interaction))
    # {
    #
    #   correction <- tf$multiply(tf$constant(matrix(addconst_interaction), dtype="float32"),
    #                             tf_row_tensor_left_part(input_theta_y, interact_pred)) %>%
    #     thetas_layer()
    #   correction_prime <- tf$multiply(tf$constant(matrix(addconst_interaction), dtype="float32"),
    #                                   tf_row_tensor_left_part(input_theta_y_prime, interact_pred)) %>%
    #     thetas_layer()
    #
    #   aTtheta <- tf$add(aTtheta, correction)
    #   aPrimeTtheta <- tf$add(aPrimeTtheta, correction_prime)
    #
    # }
    
  }
  
  # Check if model is SATs
  if(atm_lags){
    
    # if h1 is given, transform F_{t-1} using h1
    if(!is.null(interact_pred_trafo)){
      
      rho_parts <- lapply(input_theta_atm, function(inp) 
        tf_row_tensor_right_part(inp, interact_pred) %>%
          thetas_layer() %>%
          layer_lambda(f = interact_pred_trafo))
      
      
      aTtheta_lags <- lapply(rho_parts, function(rp) tf$matmul(
        tf$multiply(tf_row_tensor_left_part(input_theta_y,
                                            interact_pred),
                    rp),
        tf$ones(shape = c((order_bsp+1L)*(ncol(interact_pred)[[1]]),1))
      ))
      
      
      # otherwise just combine lags with interact_pred via RWT
      # and add thetas layer on top
    }else{ 
      
      ## thetas
      AoBs <- lapply(input_theta_atm, function(inp) 
        tf_row_tensor(inp, interact_pred))
      
      aTtheta_lags <- lapply(AoBs, function(aob) aob %>% thetas_layer())
      
      
    }
    
    # SAT with no additional h2 
    # just learn linear combination of transformed lags
    if(is.null(atm_toplayer))
      atm_toplayer <- function(x){ x %>% layer_dense(1) }
    
    aTtheta_lags <- if(length(aTtheta_lags)==1) aTtheta_lags[[1]] else layer_concatenate(aTtheta_lags)
    final_eta_pred <- final_eta_pred + (aTtheta_lags %>% atm_toplayer)
    
  } # SATs end
  
  # if(!is.null(addconst_interaction))
  # {
  #
  #   modeled_terms <- layer_concatenate(list(
  #     final_eta_pred,
  #     aTtheta,
  #     aPrimeTtheta,
  #     correction,
  #     correction_prime
  #     # tf$add(tf$multiply(tf$constant(matrix(0),dtype="float32"), aTtheta), correction)
  #   ))
  #
  # }else{
  
  modeled_terms <- layer_concatenate(list(
    final_eta_pred,
    aTtheta,
    aPrimeTtheta
  ))
  
  # }
  
  # evaluate base_distribution once
  # otherwise it will be a symbolic tensor
  if(is.null(base_distribution) || (is.character(base_distribution) & 
                                    base_distribution=="normal")){
    bd <- tfd_normal(loc = 0, scale = 1)
  }else if((is.character(base_distribution) & 
            base_distribution=="logistic")){ 
    bd <- tfd_logistic(loc = 0, scale = 1)
  }else{
    bd <- base_distribution
  }
  
  neg_ll <- function(y, model) {
    
    # shift term/lin pred
    w_eta <- model[, 1, drop = FALSE]
    
    # first part of the loglikelihood, n x (order + 1)
    aTtheta <- model[, 2, drop = FALSE]
    aTtheta_shift <- aTtheta + w_eta
    first_term <- bd %>% tfd_log_prob(aTtheta_shift)
    
    # second part of the loglikelihood
    aPrimeTtheta <- model[, 3, drop =  FALSE]
    sec_term <- tf$math$log(tf$clip_by_value(aPrimeTtheta, 1e-8, Inf))
    
    neglogLik <- -1 * (first_term + sec_term)
    
    return(neglogLik)
  }
  
  
  inputList <- unname(c(
    unlist(inputs_deep[!sapply(inputs_deep, is.null)],
           recursive = F),
    inputs_struct[!sapply(inputs_struct, is.null)]))
  
  if(!is.null(input_theta_atm))
    inputList <- unname(c(inputList, input_theta_atm))
  
  inputList <- unname(c(inputList, 
                        unlist(ox[!sapply(ox, is.null)]),
                        input_theta_y,
                        input_theta_y_prime
  )
  )
  
  model <- keras_model(inputs = inputList,
                       outputs = modeled_terms)
  
  mono_layer_ind <- grep(
    "constraint_mono_layer",
    sapply(model$trainable_weights, function(x) x$name)
  )
  
  # add penalty for interaction term
  if(is.null(list_structured[[2]]))
  {
    if(!is.null(lambda_lasso) & is.null(lambda_ridge)){
      
      reg = function(x) tf$keras$regularizers$l1(l=lambda_lasso)(
        model$trainable_weights[[mono_layer_ind]]
      )
      
    }else if(!is.null(lambda_ridge) & is.null(lambda_lasso)){
      
      reg = function(x) tf$keras$regularizers$l2(l=lambda_ridge)(
        model$trainable_weights[[mono_layer_ind]]
      )
      
      
    }else if(!is.null(lambda_ridge) & !is.null(lambda_lasso)){
      
      reg = function(x) tf$keras$regularizers$l1_l2(
        l1=lambda_lasso,
        l2=lambda_ridge)(model$trainable_weights[[mono_layer_ind]])
      
    }else{
      
      reg = NULL # no penalty
      
    }
    
    reg2 = NULL
    
  }else{
    
    if(penalize_bsp)
      bspP <- secondOrderPenBSP(order_bsp, order_diff = order_bsp_penalty)
    bigP <- list_structured[[2]]
    if(!is.null(deep_part_ia))
    {
      
      bigP <- bdiag(list(bigP, diag(rep(1, ncol(deep_part_ia)[[1]]))))
      
    }
    if(length(bigP@x)==0 & penalize_bsp==0){
      reg = NULL
    }else if(penalize_bsp==0){
      
      reg = function(x) k_mean(k_batch_dot(model$trainable_weights[[mono_layer_ind]], k_dot(
        # tf$constant(
        sparse_mat_to_tensor(as(kronecker(diag(rep(1, ncol(input_theta_y)[[1]])),bigP),
                                "CsparseMatrix")),
        # dtype = "float32"),
        model$trainable_weights[[mono_layer_ind]]),
        axes=2) # 1-based
      )
      
    }else if(length(bigP@x)==0)
    {
      
      reg = function(x) k_mean(k_batch_dot(model$trainable_weights[[mono_layer_ind]],
                                           k_dot(
                                             # tf$constant(
                                             sparse_mat_to_tensor(as(kronecker(penalize_bsp*bspP,
                                                                               diag(rep(1, ncol(interact_pred)[[1]]))),
                                                                     "CsparseMatrix")),
                                             # dtype = "float32"),
                                             model$trainable_weights[[mono_layer_ind]]),
                                           axes=2) # 1-based
      )
      
    }else{
      
      reg = function(x) k_mean(k_batch_dot(model$trainable_weights[[mono_layer_ind]],
                                           k_dot(
                                             # tf$constant(
                                             sparse_mat_to_tensor(
                                               as(kronecker(penalize_bsp*bspP, diag(rep(1, ncol(interact_pred)[[1]])))
                                                  + kronecker(diag(rep(1, ncol(input_theta_y)[[1]])),bigP), "CsparseMatrix")),
                                             # dtype = "float32"),
                                             model$trainable_weights[[mono_layer_ind]]),
                                           axes=2) # 1-based
      )
      
    }
    
    if(!is.null(lambda_lasso) & is.null(lambda_ridge)){
      
      reg2 = function(x) tf$keras$regularizers$l1(l=lambda_lasso)(
        model$trainable_weights[[mono_layer_ind]])
      
    }else if(!is.null(lambda_ridge) & is.null(lambda_lasso)){
      
      reg2 = function(x) tf$keras$regularizers$l2(l=lambda_ridge)(
        model$trainable_weights[[mono_layer_ind]])
      
      
    }else if(!is.null(lambda_ridge) & !is.null(lambda_lasso)){
      
      reg2 = function(x) tf$keras$regularizers$l1_l2(
        l1=lambda_lasso,
        l2=lambda_ridge)(model$trainable_weights[[mono_layer_ind]])
      
    }else{
      
      reg2 = NULL
      
    }
    
  }
  
  # add penalization
  if(!is.null(reg)) model$add_loss(reg)
  # add additional l1 or l2
  if(!is.null(reg2)) model$add_loss(reg2)
  
  model %>% compile(
    optimizer = optimizer,
    loss      = neg_ll,
    metrics   = monitor_metric
  )
  
  return(model)
  
  
}
