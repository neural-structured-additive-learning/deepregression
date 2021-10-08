# used by gam_processor
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
    
  }else if(length(org_values)==2){
    
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
    
  }else{
    
    warning("Plot for more than 2 dimensions not implemented yet.")
    
  }
  
  return(plotData)
  
}


layer_spline = function(units = 1L, P, name) {
  python_path <- system.file("python", package = "deepregression")
  splines <- reticulate::import_from_path("psplines", path = python_path)
  
  return(splines$layer_spline(P = as.matrix(P), units = units, name = name))
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

#' @export
extract_sp_S <- function(x)
{
  
  if((length(x[[1]]$S)==1 & is.null(x[[1]]$by.level)) | 
     (length(x[[1]]$S)==2 & !is.null(length(x[[1]]$margin)))){ 
    
    S <- x[[1]]$S 
    
  }else{
    
    S <- lapply(x,function(y) y$S[[1]])
    
  }
  
  sp <- lapply(x, "[[", "sp")
  
  return(list(sp = sp, S = S))
  
}


defaultSmoothingFun <- function(st, this_df, hat1, sp_scale, 
                                null_space_penalty, anisotropic){
  if(st[[1]]$by!="NA" && length(st)!=1)
    return(unlist(lapply(1:length(st), function(i) 
      defaultSmoothingFun(st[i], this_df = this_df, 
                          hat1 = hat1, sp_scale = sp_scale,
                          null_space_penalty = null_space_penalty,
                          anisotropic = anisotropic)), recursive = F))
  # TODO: Extend for TPs (S[[1]] is only the first matrix)
  if(length(st[[1]]$S)==1 & length(st)==1){ 
    S <- st[[1]]$S[[1]]
  }else if(length(st[[1]]$S)!=1){
    if(!anisotropic | !is.null(st[[1]]$flev)){
      S <- Reduce("+", st[[1]]$S) 
    }else{
      S <- st[[1]]$S
    }
  }else{ 
    S <- Matrix::bdiag(lapply(st,function(x)x$S[[1]]))
  }
  if(length(st)==1 & is.null(st[[1]]$margin)){ 
    X <- st[[1]]$X 
    if(is.list(S) && length(S)>1){
      if(null_space_penalty) S <- S[[1]]+S[[2]] else
        stop("Wrong dimensions of smoothing penalty matrices.")
    }
  }else{ 
    if(anisotropic){
      if(length(this_df)==1) this_df <- rep(this_df, length(st[[1]]$margin))
      st[[1]]$sp <- sapply(1:length(st[[1]]$margin), function(i)
      { 
        DRO(st[[1]]$margin[[i]]$X, 
            df = this_df[i], 
            dmat = st[[1]]$margin[[i]]$S[[1]], 
            hat1 = hat1
        )["lambda"]/sp_scale(st[[1]]$margin[[i]]$X) + 
          null_space_penalty
      })
      return(st)
    }else{
      X <- do.call("cbind", lapply(st,"[[","X"))
    }
  }
  st[[1]]$sp = DRO(X, df = this_df, dmat = S, hat1 = hat1)["lambda"] * 
    sp_scale(X) + 
    null_space_penalty
  return(st)
}

#' Function to define smoothness and call mgcv's smooth constructor
#' 
#' @param object character defining the model term
#' @param data data.frame or list
#' @param controls controls for penalization
#' @return constructed smooth term
#'
#' @export
#'
#'
handle_gam_term <- function(
  object,
  data,
  controls
)
{
  
  # check for df argument and remove
  df <- suppressWarnings(extractval(object, "df"))
  if(!is.null(df)){ 
    object <- remove_df(object)
  }else{
    df <- controls$df
  }
  names_s <- all.vars(as.formula(paste0("~", object)))
  sterm <- smoothCon(eval(parse(text=object)),
                     data=data.frame(data[names_s]),
                     absorb.cons = controls$absorb_cons,
                     null.space.penalty = controls$null_space_penalty
  )
  sterm <- controls$defaultSmoothing(sterm, df)
  return(sterm)
  
}


remove_df <- function(object)
{
  
  gsub(",\\s?df\\s?=\\s?[0-9.-]+","",object)
  
}


predict_gam_handler <- function(object, newdata)
{
  
  if(is.list(object) && length(object)==1) return(PredictMat(object[[1]], as.data.frame(newdata)))
  return(do.call("cbind", lapply(object, function(obj) PredictMat(obj, as.data.frame(newdata))))  )
  
}


get_gam_part <- function(term, specials = c("s", "te", "ti"))
{
  
  gsub("vc\\(((s|te|ti)\\(.*\\))\\,\\sby=.*\\)","\\1", term)
  
}
