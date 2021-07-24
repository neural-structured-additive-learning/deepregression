#' Control function to define the processor for terms in the formula
#' 
#' @param formula the formula to be processed
#' @param data the data for the terms in the formula
#' @param controls controls for gam terms
#' @param specials_to_oz specials that should be automatically checked for 
#' @param automatic_oz_check logical; whether to automatically check for DNNs to be orthogonalized
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
  ...){
  
  defaults <- 
    list(s = gam_processor,
         te = gam_processor,
         ti = gam_processor,
         lin = lin_processor,
         vc = vc_processor,
         lasso = l1_processor,
         ridge = l2_processor,
         offset = offset_processor,
         vi = vi_processor
    )
  
  dots <- list(...)
  
  if(is.null(names(dots)))
    stop("Please provide named arguments.")
  
  procs <- c(defaults, dots)
  specials <- names(procs)
  
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
    args$controls <- NULL 
    if(is.null(spec)){
      #args$param_nr <- paste0(args$param_nr, ".", lin_counter)
      result[[i]] <- c(list_terms[[i]], do.call(lin_processor,
                                                args))
      lin_counter <- lin_counter+1
    }else{
      if(spec %in% c("s", "te", "ti", "vc")) args$controls <- controls
      result[[i]] <- c(list_terms[[i]], do.call(procs[[spec]], args))
    }
    
  }
  
  return(result)
  
}

lin_processor <- function(term, data, output_dim, param_nr){
  
  # for all non-specials
  if(term=="(Intercept)") term <- "1"
  if(grepl("lin(.*)", term)) term <- paste0("-1 + ", paste(extractvar(term),
                                                           collapse = " + "))
  
  # model.matrix cannot deal with lists in the case if ...
  if(term=="1" & !is.data.frame(data)) 
    data <- as.data.frame(data[[1]])
  
  list(
    data_trafo = function() model.matrix(object = as.formula(paste0("~", term)), 
                                         data = data),
    predict_trafo = function(newdata){ 
      if(term==1) newdata <- as.data.frame(newdata[[1]])
      return(
        model.matrix(object = as.formula(paste0("~", term)),
                     data = as.data.frame(newdata))
      )
      },
    input_dim = ncol(model.matrix(object = as.formula(paste0("~", term)), 
                                  data = data)),
    layer = function(x, ...)
      return(
        tf$keras$layers$Dense(
          units = output_dim,
          use_bias = FALSE,
          name = makelayername(term, param_nr),
          ...)(x)),
    coef = function(weights) weights
  )
  
}

gam_processor <- function(term, data, output_dim, param_nr, controls){
  
  output_dim <- as.integer(output_dim)
  # extract mgcv smooth object
  evaluated_gam_term <- handle_gam_term(object = term, 
                                        data = data, 
                                        controls = controls)
  # check if smoothing parameter is set or not
  trainable_smooth <- is.null(evaluated_gam_term[[1]]$sp)
  # get sp and S
  sp_and_S <- extract_sp_S(evaluated_gam_term)
  # constraint
  if(controls$zero_constraint_for_smooths & 
     length(evaluated_gam_term)==1 & 
     !controls$variational){
    Z <- orthog_structured_smooths_Z(
      evaluated_gam_term[[1]]$X,
      matrix(rep(1,NROW(evaluated_gam_term[[1]]$X)), ncol=1)
    )
    sp_and_S[[2]][[1]] <- orthog_P(sp_and_S[[2]][[1]],Z)
  }
  # define layer  
  if(trainable_smooth){
    layer <- function(x, ...)
      return(
        layer_trainable_spline(units = output_dim, 
                               this_lambdas = rep(1, length(sp_and_S[[1]])),
                               this_P = tf$linalg$LinearOperatorFullMatrix(sp_and_S[[2]]), 
                               this_n = nrow(data),
                               name = makelayername(term, 
                                                    param_nr),
                               ...)(x))
  }else if(!controls$variational){
    layer <- function(x, ...)
      return(layer_spline(
                   name = makelayername(term, 
                                        param_nr),
                   P = as.matrix(bdiag(lapply(1:length(sp_and_S[[1]]), function(i) 
                     sp_and_S[[1]][[i]] * sp_and_S[[2]][[i]]))),
                   units = output_dim)(x))
  }else{ # TODO: move this to a dedicated processor
    layer <- function(x, ...)
      return(tfp$layer$DenseVariational(
        make_posterior_fn = controls$make_posterior_fn,
        make_prior_fn = prior_pspline(
          kernel_size = output_dim,
          P = bdiag(lapply(1:length(sp_and_S), function(i) sp_and_S[[1]][[i]] * sp_and_S[[2]][[i]]))
        )
      )(x))
  }
  
  list(
    data_trafo = function() evaluated_gam_term[[1]]$X %*% Z,
    predict_trafo = function(newdata) predict_gam_handler(evaluated_gam_term, newdata = newdata) %*% Z,
    input_dim = ncol(evaluated_gam_term[[1]]$X %*% Z),
    layer = layer,
    coef = function(weights) weights,
    partial_effect = function(weights, newdata=NULL){
      if(is.null(newdata))
        return(evaluated_gam_term[[1]]$X %*% Z %*% weights)
      return(predict_gam_handler(evaluated_gam_term, newdata = newdata) %*% Z %*% weights)
    },
    plot_fun = function(self, weights, grid_length) gam_plot_data(self, weights, grid_length),
    get_org_values = function() data[extractvar(term)]
  )
}

fac_processor <- function(term, data, output_dim, param_nr){
  # factor_layer
  list(
    data_trafo = function() as.integer(data[extractvar(term)]),
    predict_trafo = function(newdata) as.integer(newdata[extractvar(term)]),
    input_dim = extractlen(term, data),
    layer = function(x, ...)
      return(tf$one_hot(tf$cast(x, dtype="int32"), 
                 depth = nlevels(data[[extractvar(term)]])) %>% 
      tf$keras$layers$Dense(
        units = output_dim,
        kernel_regularizer = tf$keras$regularizers$l2(l = extractval(term, "la")),
        name = makelayername(term, 
                             param_nr),
        ...
        )),
    coef = function(weights) weights
  )
}
  
vc_processor <- function(term, data, output_dim, param_nr, controls){
  # vc (old: vc, vcc)
  vars <- extractvar(term)
  byt <- form2text(extractval(term, "by"))
  # extract gam part
  gampart <- get_gam_part(term)
  if(length(setdiff(vars, c(extractvar(gampart), extractvar(byt))))>0)
    stop("vc terms currently only suppoert one gam term and one by term.")
  
  nlev <- sapply(data[extractvar(byt)], nlevels)
  
  output_dim <- as.integer(output_dim)
  # extract mgcv smooth object
  evaluated_gam_term <- handle_gam_term(object = gampart, 
                                        data = data, 
                                        controls = controls)
  
  ncolNum <- ncol(evaluated_gam_term[[1]]$X)
  
  P <- evaluated_gam_term[[1]]$S[[1]]
  
  if(length(nlev)==1){
    layer <- vc_block(ncolNum, nlev, penalty = P, 
                      name = makelayername(term, param_nr), units = units)
  }else if(length(nlev)==2){
    layer <- vc_block(ncolNum, nlev[1], nlev[2], penalty = P, 
                      name = makelayername(term, param_nr), units = units)
  }else{
    stop("vc terms with more than 2 factors currently not supported.")
  }

  list(
    data_trafo = function() do.call("cbind", c(evaluated_gam_term[[1]]$X, 
                                               as.integer(data[byt]))),
    predict_trafo = function(newdata) do.call("cbind", c(
      predict_gam_handler(evaluated_gam_term, newdata = newdata),
      as.integer(data[byt]))),
    input_dim = ncolNum + length(nlev),
    layer = layer,
    coef = function(weights) weights
  )
}

l1_processor <- function(term, data, output_dim, param_nr){
  # l1 (Tib)
  list(
    data_trafo = function() data[extractvar(term)],
    predict_trafo = function(newdata) newdata[extractvar(term)],
    input_dim = extractlen(term, data),
    layer = function(x, ...) 
      return(tib_layer(
        units = as.integer(output_dim),
        la = extractval(term, "la"),
        name = makelayername(term, 
                             param_nr),
        ...
      )(x)),
    coef = function(weights) weights[[1]] * matrix(rep(weights[[2]], each=ncol(weights[[1]])), 
                                                   ncol=ncol(weights[[1]]), byrow = TRUE)
  )
  
}

l2_processor <- function(term, data, output_dim, param_nr){
  # ridge
  list(
    data_trafo = function() data[extractvar(term)],
    predict_trafo = function(newdata) newdata[extractvar(term)],
    input_dim = extractlen(term, data),
    layer = function(x, ...)
      return(tf$keras$layers$Dense(units = output_dim, 
                                   kernel_regularizer = 
                                     tf$keras$regularizers$l2(l = extractval(term, "la")),
                                   name = makelayername(term, 
                                                        param_nr),
                                   ...)(x)),
    coef = function(weights) weights
  )
  
}
  
offset_processor <- function(term, data, output_dim, param_nr){
  # offset
  list(
    data_trafo = function() data[extractvar(term)],
    predict_trafo = function(newdata) newdata[extractvar(term)],
    input_dim = extractlen(term, data),
    layer = function(x, ...)
      return(tf$keras$layers$Dense(units = 1L,
                                   trainable = FALSE,
                                   kernel_initializer = tf$keras$initializers$Ones,
                                   name = makelayername(term, 
                                                        param_nr),
                                   ...)(x))
  )
}

vi_processor <- function(term, data, output_dim, param_nr){
  # vi_layer
  list(
    data_trafo = function() data[extractvar(term)],
    predict_trafo = function(newdata) newdata[extractvar(term)],
    input_dim = extractlen(term, data),
    layer = function(x, ...) 
      return(
        tfp$layers$DenseVariational(
          input_dim = extractlen(term),
          units = as.integer(output_dim),
          name = makelayername(term, 
                               param_nr),
          make_posterior_fn = evalarg(term, "posterior"),
          make_prior_fn = evalarg(term, "prior"),
          kl_weight = evalarg(term, "kl_weight"),
          ...)(x))
  )
}

dnn_placeholder_processor <- function(dnn){
  function(term, data, output_dim, param_nr){
    list(
      data_trafo = function() data[extractvar(term)],
      predict_trafo = function(newdata) newdata[extractvar(term)],
      input_dim = extractlen(term, data),
      layer = dnn
    )
  }
}

#### helper functions ####

makelayername <- function(term, param_nr)
{
  
  if(class(term)=="formula") term <- form2text(term)
  return(paste0(make_valid_layername(term), "_", param_nr))
  
}

extractvar <- function(term)
{
  
  all.vars(as.formula(paste0("~", term)))
  
}

extractval <- function(term, name)
{
  
  if(is.character(term)) term <- as.formula(paste0("~", term))
  inputs <- as.list(as.list(term)[[2]])[-1]
  if(name %in% names(inputs)) return(inputs[[name]])
  warning("Argument ", name, " not found. Setting it to some default.")
  if(name=="df") return(NULL) else if(name=="la") return(0.1) else return(1)

}

extractlen <- function(term, data)
{
  
  vars <- extractvar(term)
  sum(sapply(vars, function(v) NCOL(data[v])))
  
}

evalarg <- function(term)
{
  
  
  
}

get_gam_part <- function(term, specials = c("s", "te", "ti"))
{
  
  gsub("vc\\(((s|te|ti)\\(.*\\))\\,\\sby=.*\\)","\\1", term)
  
}

form2text <- function(form)
{
  
  return(gsub(" ","", (Reduce(paste, deparse(form)))))
  
}

get_special <- function(term, specials)
{
  
  sp <- attr(terms.formula(as.formula(paste0("~",term)), 
                           specials = specials), "specials")
  names(unlist(sp))
  
}

predict_gam_handler <- function(object, newdata)
{

  if(is.list(object) && length(object)==1) return(PredictMat(object[[1]], as.data.frame(newdata)))
  return(lapply(object, function(obj) PredictMat(obj, newdata)))  
  
}

get_names_pfc <- function(pfc) sapply(pfc, "[[", "term")
