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
      defaultSmoothing(st[i], this_df = this_df)), recursive = F))
  # TODO: Extend for TPs (S[[1]] is only the first matrix)
  if(length(st[[1]]$S)==1 & length(st)==1){ 
    S <- st[[1]]$S[[1]]
  }else if(length(st[[1]]$S)!=1){
    if(!anisotropic){
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
  st[[1]]$sp = DRO(X, df = this_df, dmat = S, hat1 = hat1)["lambda"]/
    sp_scale(X) + 
    null_space_penalty
  return(st)
}

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
  
  gsub(",\\s?df\\s?=\\s?[0-9]*","",object)
  
}
