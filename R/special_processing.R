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
#' @param parsing_options options 
#' @param ... further processors
#' @return returns a processor function
#' 
#' 
process_terms <- function(
  form, data, controls, 
  output_dim, param_nr, 
  parsing_options,
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
         grlasso = l21_processor,
         ridge = l2_processor,
         offsetx = offset_processor,
         rwt = rwt_processor,
         const = const_broadcasting_processor,
         mult = multiply_processor
    )
  
  dots <- list(...)
  
  if(length(dots)>0 && is.null(names(dots)))
    stop("Please provide named arguments.")
  
  procs <- c(defaults, dots)
  specials <- names(procs)
  specials <- specials[sapply(specials, nchar)>0]
  
  # otherwise offset is dropped
  if(parsing_options$check_form) form <- rename_offset(form)
  
  # for row-wise tensor product
  if(parsing_options$check_form) form <- rename_rwt(form)
  
  list_terms <- separate_define_relation(form = form, 
                                         specials = specials, 
                                         specials_to_oz = specials_to_oz, 
                                         automatic_oz_check = automatic_oz_check,
                                         simplify = !parsing_options$check_form)
  
  if("1" %in% sapply(list_terms, "[[", "term"))
  {
    list_terms[[which(sapply(list_terms, "[[", "term")=="1")]]$term <- 
      "(Intercept)"
  }
  
  args <- list(data = data, output_dim = output_dim, param_nr = param_nr)
  result <- list()
  
  # add intercept terms
  if(parsing_options$check_form)
    if(attr(terms.formula(form), "intercept") & !"(Intercept)" %in% 
       sapply(list_terms, "[[", "term"))
      list_terms[[length(list_terms)+1]] <- 
    list(term = "(Intercept)",
         nr = length(list_terms)+1,
         left_from_oz = TRUE,
         right_from_oz = NULL)
  
  for(i in 1:length(list_terms)){
    
    args$term = list_terms[[i]]$term
    spec <- get_special(list_terms[[i]]$term, specials = specials, 
                        simplify = !parsing_options$check_form)
    args$controls <- controls 
    args$controls$procs <- procs
    if(is.null(spec)){
      if(args$term=="(Intercept)")
        result[[i]] <- c(list_terms[[i]], do.call(int_processor, args)) else
          result[[i]] <- c(list_terms[[i]], do.call(lin_processor, args))
    }else{
      result[[i]] <- c(list_terms[[i]], do.call(procs[[spec]], args))
    }
    
  }
  
  if(!is.null(controls$weight_options$shared_layers)){
    
    names_res <- get_names_pfc(result)
    for(group in controls$weight_options$shared_layers){
      for(term in group){
       result[[which(term==names_res)]]$shared_name <-  
         paste0("shared_", 
                makelayername(paste(group, collapse="_"), 
                              param_nr))
      }
    }
  }
  
  return(result)
  
}

#' Function that creates layer for each processor
#' 
#' @param term character; term in the formula
#' @param output_dim integer; number of units in the layer
#' @param param_nr integer; identifier for models with more 
#' than one additive predictor
#' @param controls list; control arguments which allow
#' to pass further information
#' @param layer_class a tf or keras layer function
#' @param without_layer function to be used as 
#' layer if \code{controls$with_layer} is FALSE
#' @param name character; name of layer. 
#' if NULL, \code{makelayername} will be used to create layer name
#' @param further_layer_args named list; further arguments passed to
#' the layer
#' @param layer_args_names character vector; if NULL, default
#' layer args will be used. Needs to be set for layers that do not
#' provide the arguments of a default Dense layer.
#' @param units integer; number of units for layer
#' @param ... other keras layer parameters
#' 
#' @return a basic processor list structure
#' 
#' @export
#' 
layer_generator <- function(term, output_dim, param_nr, controls, 
                            layer_class = tf$keras$layers$Dense,
                            without_layer = tf$identity,
                            name = makelayername(term, param_nr), 
                            further_layer_args = NULL,
                            layer_args_names = NULL,
                            units = as.integer(output_dim),
                            ...
                            ){
  
  const_broadcasting <- !is.null(controls$const_broadcasting) && (
    controls$const_broadcasting & output_dim>1)
  
  layer_args <- controls$weight_options$general
  layer_args <- c(layer_args, list(...))
  
  specific_opt <- term %in% names(controls$weight_options$specific)
  if(specific_opt){
    
    spop <- controls$weight_options$specific[[term]]
    layer_args[names(spop)] <- spop
    
  }
  
  
  warmstart <- term %in% names(controls$weight_options$warmstarts)
  
  if(warmstart)
    layer_args$kernel_initializer <- tf$keras$initializers$Constant(
      controls$weight_options$warmstarts[[term]]
    )
  

  if(!const_broadcasting) layer_args$units <- units else 
    layer_args$units <- controls$const_broadcasting
  layer_args$name <- name

  if(!is.null(further_layer_args)) 
    layer_args <- c(layer_args, further_layer_args)
  if(!is.null(layer_args_names)) 
    layer_args <- layer_args[layer_args_names]
  
  if(controls$with_layer){
    
    if(!const_broadcasting){
      layer = function(x){
        return(
          do.call(layer_class, layer_args)(x)
        )
      }
    }else{
      layer = function(x){
        layer_prev <- do.call(layer_class, layer_args)(x)
        return(
          tf$tile(layer_prev, multiples = list(tf$shape(layer_prev)[[1]], output_dim))
        )
      }
    }
    
  }else{
    
    layer = without_layer
    
  }
  
  return(layer)
  
  
}

int_processor <- function(term, data, output_dim, param_nr, controls){
  
  if(term=="(Intercept)") term <- "1"
  data <- as.data.frame(data[[1]])
  
  layer <- layer_generator(term = term, 
                           output_dim = output_dim, 
                           param_nr = param_nr, 
                           controls = controls)

  list(
    data_trafo = function() matrix(rep(1, nrow(data)), ncol=1),
    predict_trafo = function(newdata){ 
      return(
        matrix(rep(1, nrow(as.data.frame(newdata[[1]]))), ncol=1)
      )
    },
    input_dim = 1L,
    layer = layer,
    coef = function(weights)  as.matrix(weights),
    penalty = NULL
  )
  
  
}

lin_processor <- function(term, data, output_dim, param_nr, controls){
  
  
  layer <- layer_generator(term = term, 
                           output_dim = output_dim, 
                           param_nr = param_nr, 
                           controls = controls)
  
  if(grepl("lin(.*)", term)) term <- paste(extractvar(term, allow_ia = TRUE),
                                           collapse = " + ")
  
  data_trafo <- function(indata = data)
  {
    if(attr(terms.formula(as.formula(paste0("~", term))), "intercept")==0){
      model.matrix(object = as.formula(paste0("~", term)), 
                   data = indata)
    }else{
      model.matrix(object = as.formula(paste0("~ 1 + ", term)), 
                   data = indata)[,-1,drop=FALSE]
    }
  }
  
  list(
    data_trafo = function() data_trafo(),
    predict_trafo = function(newdata){ 
      return(
        data_trafo(as.data.frame(newdata))
      )
    },
    input_dim = as.integer(ncol(data_trafo())),
    layer = layer,
    coef = function(weights)  as.matrix(weights),
    penalty = NULL
  )
  
}

gam_processor <- function(term, data, output_dim, param_nr, controls){
  
  output_dim <- as.integer(output_dim)
  # extract mgcv smooth object
  P <- create_P(get_gamdata(term, param_nr, controls$gamdata, what="sp_and_S"),
                controls$sp_scale(data))
  
  layer <- layer_generator(term = term, 
                           output_dim = output_dim, 
                           param_nr = param_nr, 
                           controls = controls,
                           further_layer_args = list(P = P),
                           layer_args_names = c("name", "units", "P", "trainable", 
                                                "kernel_initializer"),
                           layer_class = layer_spline
                           )

  list(
    data_trafo = get_gamdata(term, param_nr, controls$gamdata, what="data_trafo"),
    predict_trafo = get_gamdata(term, param_nr, controls$gamdata, what="predict_trafo"),
    input_dim = get_gamdata(term, param_nr, controls$gamdata, what="input_dim"),
    layer = layer,
    coef = function(weights)  as.matrix(weights),
    partial_effect = get_gamdata(term, param_nr, controls$gamdata, what="partial_effect"),
    plot_fun = function(self, weights, grid_length) gam_plot_data(self, weights, grid_length),
    get_org_values = function() data[extractvar(term)],
    penalty = list(type = "spline", values = P, dim = output_dim),
    gamdata_nr = get_gamdata_reduced_nr(term, param_nr, controls$gamdata),
    gamdata_combined = FALSE
  )
}


l1_processor <- function(term, data, output_dim, param_nr, controls){
  # l1 (Tib)
  lambda = controls$sp_scale(data) * as.numeric(extractval(term, "la"))
  
  layer <- layer_generator(term = term, 
                           output_dim = output_dim, 
                           param_nr = param_nr, 
                           controls = controls,
                           further_layer_args = list(la = lambda),
                           layer_args_names = c("name", "units", "la"),
                           layer_class = tib_layer,
                           without_layer = function(x, ...) 
                             return(simplyconnected_layer(
                               la = lambda,
                               name = makelayername(term, param_nr),
                               ...
                             )(x))
  )

  penalty <- if(output_dim > 1){
    list(type = "inverse_group", values = lambda, dim = output_dim)
  }else{
    list(type = "l1", values = lambda, dim = output_dim)
  }
  
  list(
    data_trafo = function() data[extractvar(term)],
    predict_trafo = function(newdata) newdata[extractvar(term)],
    input_dim = as.integer(extractlen(term, data)),
    layer = layer,
    coef = function(weights){ 
      weights <- lapply(weights, as.matrix)
      return(
        weights[[1]] * matrix(rep(weights[[2]], each=ncol(weights[[1]])), 
                              ncol=ncol(weights[[1]]), byrow = TRUE)
      )
    },
    penalty = penalty
  )
  
}

l21_processor <- function(term, data, output_dim, param_nr, controls){
  
  lambda = controls$sp_scale(data) * as.numeric(extractval(term, "la"))
  
  layer <- layer_generator(term = term, 
                           output_dim = output_dim, 
                           param_nr = param_nr, 
                           controls = controls,
                           further_layer_args = list(group_idx = NULL, la = lambda),
                           layer_args_names = c("name", "units", "la", "group_idx"),
                           layer_class = tibgroup_layer,
  )
  
  data_trafo <- function(indata = data) 
    model.matrix(object = as.formula(paste0("~ 1 + ", extractvar(term))), 
                 data = indata)[,-1,drop=FALSE]
  
  list(
    data_trafo = function() data_trafo(),
    predict_trafo = function(newdata){ 
      return(
        data_trafo(as.data.frame(newdata))
      )
    },
    input_dim = as.integer(ncol(data_trafo())),
    layer = layer,
    coef = function(weights){ 
      weights <- lapply(weights, as.matrix)
      return(
        weights[[1]][[1]] * weights[[2]]
      )
    },
    penalty = list(type = "l1group", values = lambda, dim = output_dim)
  )
  
}


l2_processor <- function(term, data, output_dim, param_nr, controls){
  # ridge
  
  lambda = controls$sp_scale(data) * extractval(term, "la")
  
  layer <- layer_generator(term = term, 
                           output_dim = output_dim, 
                           param_nr = param_nr, 
                           controls = controls,
                           kernel_regularizer = tf$keras$regularizers$l2(l = lambda)
  )

  list(
    data_trafo = function() data[extractvar(term)],
    predict_trafo = function(newdata) newdata[extractvar(term)],
    input_dim = as.integer(extractlen(term, data)),
    layer = layer,
    coef = function(weights)  as.matrix(weights),
    penalty = list(type = "l2", values = lambda, dim = output_dim)
  )
  
}

offset_processor <- function(term, data, output_dim, param_nr, controls=NULL){
  
  layer <- layer_generator(term = term, 
                           output_dim = output_dim, 
                           param_nr = param_nr, 
                           controls = controls,
                           trainable = FALSE,
                           kernel_initializer = tf$keras$initializers$Ones
                           )
  
  list(
    data_trafo = function() data[extractvar(term)],
    predict_trafo = function(newdata) newdata[extractvar(term)],
    input_dim = as.integer(extractlen(term, data)),
    layer = layer
  )
}

rwt_processor <- function(term, data, output_dim, param_nr, controls){
  
  special_layer <- suppressWarnings(extractval(term, "layer"))
  if(!is.null(special_layer)) special_layer <- as.character(special_layer)
  term <- remove_layer(term)
  
  terms <- get_terms_rwt(term)
  
  terms <- lapply(terms, function(t){ 
    args <- list(term = t, data = data, output_dim = output_dim,
                 param_nr = param_nr, controls = controls)
    args$controls$with_layer <- FALSE
    spec <- get_special(t, specials = names(controls$procs))
    if(is.null(spec)){
      if(t=="1")
        return(do.call(int_processor, args)) else
          return(do.call(lin_processor, args))
    }
    do.call(controls$procs[[spec]], args)
  })
  
  dims <- sapply(terms, "[[", "input_dim")
  penalties <- lapply(terms, "[[", "penalty")
  combined_penalty <- combine_penalties(penalties, dims)
  
  if(is.null(special_layer)){ 
    
    this_layer <- function(...)
      tf$keras$layers$Dense(units = output_dim,
                            trainable = TRUE,
                            use_bias = FALSE,
                            kernel_regularizer = combined_penalty,
                            name = makelayername(term, param_nr),
                            ...) 
    
  }else{
    
    # special_layer <- special_layer[!sapply(special_layer, is.null)]
    # if(length(special_layer)==2) 
      # stop("In an RWT, only a single term can have a special layer.")
    
    this_layer <- function(...){
      
      args <- c(list(
        units = output_dim,
        kernel_regularizer = combined_penalty,
        name = makelayername(term, param_nr),
        ...
      ), controls$special_layer_args)
      
      do.call(special_layer, args)
      
    }
    
  }
  
  if(controls$with_layer){
    layer = function(x, ...){
      a <- tf_stride_cols(x, 1L, as.integer(dims[1]))
      b <- tf_stride_cols(x, 1L + as.integer(dims[1]), as.integer(sum(dims)))
      rwt <- tf_row_tensor(a,b)
      return(this_layer(...)(rwt))
    }
  }else{
    layer = tf$identity
  }
  
  list(
    data_trafo = function() do.call("cbind", lapply(terms, function(x) x$data_trafo())),
    predict_trafo = function(newdata) 
      do.call("cbind", lapply(terms, function(x) x$predict_trafo(newdata))),
    input_dim = sum(dims),
    layer = layer,
    coef = function(weights) lapply(terms, function(x) x$coef(weights)),
    partial_effect = function(...) lapply(terms, function(x) x$partial_effect(...)),
    plot_fun = function(...) lapply(terms, function(x) x$plot_fun(...)),
    get_org_values = function() do.call("cbind", lapply(terms, function(x) x$get_org_values())),
    penalty = penalties
  )
  
}

multiply_processor <- function(term, data, output_dim, param_nr, controls){
  
  terms <- get_terms_mult(term)
  
  terms <- lapply(terms, function(t){ 
    args <- list(term = t, data = data, output_dim = output_dim,
                 param_nr = param_nr, controls = controls)
    spec <- get_special(t, specials = names(controls$procs))
    if(is.null(spec)){
      if(t=="1")
        return(do.call(int_processor, args)) else
          return(do.call(lin_processor, args))
    }
    do.call(controls$procs[[spec]], args)
  })
  
  dims <- sapply(terms, "[[", "input_dim")
  csd <- c(0, cumsum(dims))
  penalties <- lapply(terms, "[[", "penalty")

  layer = function(x, ...){
    
    inps <- lapply(1:length(dims), function(i){
      tf_stride_cols(x, as.integer(csd[i]+1), as.integer(csd[i+1]))
    })
    outp <- lapply(1:length(inps), function(i) terms[[i]]$layer(inps[[i]]))
    return(tf$keras$layers$multiply(outp))
  }
  
  list(
    data_trafo = function() 
      do.call("cbind", lapply(terms, function(x) to_matrix(x$data_trafo()))),
    predict_trafo = function(newdata) 
      do.call("cbind", lapply(terms, function(x) to_matrix(x$predict_trafo(newdata)))),
    input_dim = sum(dims),
    layer = layer,
    coef = function(weights) lapply(terms, function(x) if(!is.null(x$coef)) x$coef(weights)),
    partial_effect = function(...) lapply(terms, function(x) x$partial_effect(...)),
    plot_fun = function(...) lapply(terms, function(x) x$plot_fun(...)),
    get_org_values = function() do.call("cbind", lapply(terms, function(x) x$get_org_values())),
    penalty = penalties
  )
  
}

const_broadcasting_processor <- function(term, data, output_dim, param_nr, controls){
  
  #controls$const_broadcasting <- 
  # as.integer(extractval(term, name="dim", TRUE, 1L))
  term <- gsub("const\\((.*)\\)", "\\1", term)
  
  spec <- get_special(term, specials = names(controls$procs))
  
  args <- list(data = data, output_dim = 1L, param_nr = param_nr)
  args$term <- term
  args$controls <- controls 

  if(is.null(spec)){
    if(args$term=="1")
      ret <- c(term, do.call(int_processor, args)) else
        ret <- c(term, do.call(lin_processor, args))
  }else{
    ret <- c(term, do.call(procs[[spec]], args))
  }
  
  ret$output_dim = output_dim
  layer_ft <- ret$layer
  ret$layer <- function(x){
    
    return(
      layer_ft(x) %>% layer_dense(units = as.integer(output_dim), 
                                  activation = "linear", 
                                  use_bias = FALSE, 
                                  kernel_initializer = "ones",
                                  trainable = FALSE)
    )
    
  }
  
  return(ret)
  
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

