#' Control function to define the processor for terms in the formula
#' 
#' @param form the formula to be processed
#' @param data the data for the terms in the formula
#' @param controls controls for gam terms
#' @param output_dim the output dimension of the response
#' @param specials_to_oz specials that should be automatically checked for 
#' @param automatic_oz_check logical; whether to automatically check for DNNs to be orthogonalized
#' @param identify_intercept logical; whether to make the intercept automatically identifiable
#' @param param_nr integer; identifier for the distribution parameter
#' @param ... further processors
#' @return returns a processor function
#' 
#' 
processor <- function(
  form, data, controls, 
  output_dim, param_nr, 
  specials_to_oz = c(), 
  automatic_oz_check = TRUE, 
  identify_intercept = FALSE,
  ...){
  
  defaults <- 
    list(s = gam_processor,
         te = gam_processor,
         ti = gam_processor,
         int = int_processor,
         lin = lin_processor,
         lasso = l1_processor,
         ridge = l2_processor,
         offset = offset_processor
    )
  
  dots <- list(...)
  
  if(length(dots)>0 && is.null(names(dots)))
    stop("Please provide named arguments.")
  
  procs <- c(defaults, dots)
  specials <- names(procs)
  specials <- specials[sapply(specials, nchar)>0]
  
  list_terms <- separate_define_relation(form = form, 
                                         specials = specials, 
                                         specials_to_oz = specials_to_oz, 
                                         automatic_oz_check = automatic_oz_check)
  
  if("1" %in% sapply(list_terms, "[[", "term"))
  {
    list_terms[[which(sapply(list_terms, "[[", "term")=="1")]]$term <- 
      "(Intercept)"
  }
  
  args <- list(data = data, output_dim = output_dim, param_nr = param_nr)
  result <- list()
  
  # add intercept terms
  if(attr(terms.formula(form), "intercept") & !"(Intercept)" %in% 
     sapply(list_terms, "[[", "term"))
    list_terms[[length(list_terms)+1]] <- 
    list(term = "(Intercept)",
         nr = length(list_terms)+1,
         left_from_oz = TRUE,
         right_from_oz = NULL)
  
  for(i in 1:length(list_terms)){
    
    lin_counter <- 1
    args$term = list_terms[[i]]$term
    spec <- get_special(list_terms[[i]]$term, specials = specials)
    args$controls <- controls 
    if(is.null(spec)){
      if(args$term=="(Intercept)")
        result[[i]] <- c(list_terms[[i]], do.call(int_processor, args)) else
          result[[i]] <- c(list_terms[[i]], do.call(lin_processor, args))
        lin_counter <- lin_counter+1
    }else{
      result[[i]] <- c(list_terms[[i]], do.call(procs[[spec]], args))
    }
    
  }
  
  return(result)
  
}

int_processor <- function(term, data, output_dim, param_nr, controls){
  
  if(term=="(Intercept)") term <- "1"
  data <- as.data.frame(data[[1]])
  
  if(controls$with_layer){
    layer = function(x, ...)
      return(
        tf$keras$layers$Dense(
          units = output_dim,
          use_bias = FALSE,
          name = makelayername(term, param_nr),
          ...)(x))
  }else{
    layer = tf$identity
  }
  
  list(
    data_trafo = function() matrix(rep(1, nrow(data)), ncol=1),
    predict_trafo = function(newdata){ 
      return(
        matrix(rep(1, nrow(as.data.frame(newdata[[1]]))), ncol=1)
      )
    },
    input_dim = 1L,
    layer = layer,
    coef = function(weights)  as.matrix(weights)
  )
  
  
}

lin_processor <- function(term, data, output_dim, param_nr, controls){
  
  
  if(grepl("lin(.*)", term)) term <- paste0(paste(extractvar(term),
                                                  collapse = " + "),
                                            "+ 0 ")
  
  if(controls$with_layer){
    layer = function(x, ...)
      return(
        tf$keras$layers$Dense(
          units = output_dim,
          use_bias = FALSE,
          name = makelayername(term, param_nr),
          ...)(x))
  }else{
    layer = tf$identity
  }
  
  list(
    data_trafo = function() model.matrix(object = as.formula(paste0("~ -1 + ", term)), 
                                         data = data),
    predict_trafo = function(newdata){ 
      return(
        model.matrix(object = as.formula(paste0("~ -1 + ", term)),
                     data = as.data.frame(newdata))
      )
    },
    input_dim = as.integer(ncol(model.matrix(object = as.formula(paste0("~ -1 +", term)), 
                                  data = data))),
    layer = layer,
    coef = function(weights)  as.matrix(weights)
  )
  
}

gam_processor <- function(term, data, output_dim, param_nr, controls){
  
  output_dim <- as.integer(output_dim)
  # extract mgcv smooth object
  evaluated_gam_term <- handle_gam_term(object = term, 
                                        data = data, 
                                        controls = controls)
  # get sp and S
  sp_and_S <- extract_sp_S(evaluated_gam_term)
  # extract Xs
  if(length(evaluated_gam_term)==1){
    thisX <- evaluated_gam_term[[1]]$X
  }else{
    thisX <- do.call("cbind", lapply(evaluated_gam_term, "[[", "X"))
  }
  # get default Z matrix, which is possibly overwritten afterwards
  Z <- diag(rep(1,ncol(thisX)))
  # constraint
  if(controls$zero_constraint_for_smooths & 
     length(evaluated_gam_term)==1 & 
     !evaluated_gam_term[[1]]$dim>1){
    Z <- orthog_structured_smooths_Z(
      evaluated_gam_term[[1]]$X,
      matrix(rep(1,NROW(evaluated_gam_term[[1]]$X)), ncol=1)
    )
    sp_and_S[[2]][[1]] <- orthog_P(sp_and_S[[2]][[1]],Z)
  }else if(evaluated_gam_term[[1]]$dim>1 & 
           length(evaluated_gam_term)==1){
    # tensor product -> merge and keep dummy
    sp_and_S <- list(sp = 1, 
                     S = list(do.call("+", lapply(1:length(sp_and_S[[2]]), function(i)
                       sp_and_S[[1]][[1]][i] * sp_and_S[[2]][[i]]))))
  }
  # define layer  
  if(controls$with_layer){
    layer = function(x, ...)
      return(layer_spline(
        name = makelayername(term, 
                             param_nr),
        P = as.matrix(bdiag(lapply(1:length(sp_and_S[[1]]), function(i) 
          controls$sp_scale(data) * sp_and_S[[1]][[i]] * sp_and_S[[2]][[i]]))),
        units = output_dim)(x))
  }else{
    layer = tf$identity
  }

  list(
    data_trafo = function() thisX %*% Z,
    predict_trafo = function(newdata) predict_gam_handler(evaluated_gam_term, newdata = newdata) %*% Z,
    input_dim = as.integer(ncol(thisX %*% Z)),
    layer = layer,
    coef = function(weights)  as.matrix(weights),
    partial_effect = function(weights, newdata=NULL){
      if(is.null(newdata))
        return(thisX %*% Z %*% weights)
      return(predict_gam_handler(evaluated_gam_term, newdata = newdata) %*% Z %*% weights)
    },
    plot_fun = function(self, weights, grid_length) gam_plot_data(self, weights, grid_length),
    get_org_values = function() data[extractvar(term)]
  )
}


l1_processor <- function(term, data, output_dim, param_nr, controls){
  # l1 (Tib)
  list(
    data_trafo = function() data[extractvar(term)],
    predict_trafo = function(newdata) newdata[extractvar(term)],
    input_dim = as.integer(extractlen(term, data)),
    layer = function(x, ...) 
      return(tib_layer(
        units = as.integer(output_dim),
        la = controls$sp_scale(data) * extractval(term, "la"),
        name = makelayername(term, 
                             param_nr),
        ...
      )(x)),
    coef = function(weights){ 
      weights <- lapply(weights, as.matrix)
      return(
        weights[[1]] * matrix(rep(weights[[2]], each=ncol(weights[[1]])), 
                              ncol=ncol(weights[[1]]), byrow = TRUE)
      )
    }
  )
  
}

l2_processor <- function(term, data, output_dim, param_nr, controls){
  # ridge
  
  if(controls$with_layer){
    layer = function(x, ...)
      return(tf$keras$layers$Dense(units = output_dim, 
                                   kernel_regularizer = 
                                     tf$keras$regularizers$l2(
                                       l = controls$sp_scale(data) * 
                                         extractval(term, "la")),
                                   use_bias = FALSE,
                                   name = makelayername(term, 
                                                        param_nr),
                                   ...)(x))
  }else{
    layer = tf$identity
  }
  
  list(
    data_trafo = function() data[extractvar(term)],
    predict_trafo = function(newdata) newdata[extractvar(term)],
    input_dim = as.integer(extractlen(term, data)),
    layer = layer,
    coef = function(weights)  as.matrix(weights)
  )
  
}

offset_processor <- function(term, data, output_dim, param_nr, controls=NULL){
  # offset
  list(
    data_trafo = function() data[extractvar(term)],
    predict_trafo = function(newdata) newdata[extractvar(term)],
    input_dim = as.integer(extractlen(term, data)),
    layer = function(x, ...)
      return(tf$keras$layers$Dense(units = 1L,
                                   trainable = FALSE,
                                   use_bias = FALSE,
                                   kernel_initializer = tf$keras$initializers$Ones,
                                   name = makelayername(term, 
                                                        param_nr),
                                   ...)(x))
  )
}

dnn_processor <- function(dnn){
  
  if(is.list(dnn) & length(dnn)==2){
    do.call("dnn_image_placeholder_processor", dnn)
  }else{
    dnn_placeholder_processor(dnn)
  }
}

dnn_placeholder_processor <- function(dnn){
  function(term, data, output_dim, param_nr, controls=NULL){
    list(
      data_trafo = function() data[extractvar(term)],
      predict_trafo = function(newdata) newdata[extractvar(term)],
      input_dim = as.integer(extractlen(term, data)),
      layer = dnn
    )
  }
}

dnn_image_placeholder_processor <- function(dnn, size){
  function(term, data, output_dim, param_nr, controls=NULL){
    list(
      data_trafo = function() as.data.frame(data[extractvar(term)]),
      predict_trafo = function(newdata) as.data.frame(newdata[extractvar(term)]),
      input_dim = as.integer(size),
      layer = dnn
    )
  }
}

