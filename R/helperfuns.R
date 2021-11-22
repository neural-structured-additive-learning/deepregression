is_equal_not_null <- function(x,y)
{
  
  if(is.null(y)) return(FALSE) else return(x==y)
  
}


# convert sparse matrix to sparse tensor
sparse_mat_to_tensor <- function(X)
{

  missing_ind <- setdiff(c("i","j","p"), slotNames(X))
  if(missing_ind=="j")
    j = findInterval(seq(X@x)-1,X@p[-1])
  if(missing_ind=="i") stop("Sparse Matrix with missing i not implemented yet.")
  i = X@i
  tf$SparseTensor(indices = lapply(1:length(i), function(ind) c(i[ind], j[ind])),
                  values = X@x,
                  dense_shape = as.integer(X@Dim))

}

NCOL0 <- function(x)
{
  if(is.null(x))
    return(0)
  return(NCOL(x))
}


fac_to_int_representation <- function(data)
{
  
  whfac <- sapply(data, is.factor)
  if(all(!whfac)) return(data)
  faclist <- lapply(data[which(whfac)], function(x) list(nlev=nlevels(x), lev = levels(x)))
  names(faclist) <- names(whfac[whfac])
  data[whfac] <- lapply(data[whfac], function(x) as.integer(x)-1L)
  attr(data, "faclist") <- faclist
  return(data)
  
}


subset_array <- function(x, index)
{

  # if(class(x)[1]=="placeholder") return(x[index])
  dimx <- dim(x)
  if(is.null(dimx)) dimx = 1
  tryCatch(
    eval(parse(text=paste0("x[index",
                           paste(rep(",", length(dimx)-1),collapse=""),
                           ",drop=FALSE]"))),
    error = function(e) 
      eval(parse(text=paste0("tf$constant(as.matrix(x)[index",
                             paste(rep(",", length(dimx)-1),collapse=""),
                             ",drop=FALSE], 'float32')")))
  )
}

subset_input_cov <- function(x, index)
{
  
  if(is.list(x)) lapply(x, subset_input_cov, index = index) else
    subset_array(x, index = index)
  
}

# nrow for list
nROW <- function(x)
{
  NROW(x[[1]])
}

nCOL <- function(x)
{
  if(!is.null(attr(x, "dims"))) return(attr(x, "dims")[-1])
  lapply(x, function(y) if(is.null(dim(y))) 1 else dim(y)[-1])
}

nestNCOL <- function(x)
{

  res <- list()
  for(i in 1:length(x)){

    if(is.list(x[[i]]) & length(x[[i]])>=1 & !is.null(x[[i]][[1]])){
      res[[i]] <- nestNCOL(x[[i]])
    }else if((is.list(x[[i]]) & length(x[[i]])==0) | is.null(x[[i]][[1]])){
      res[[i]] <- 0
    }else{
      res[[i]] <- NCOL(x[[i]])
    }

  }

  return(res)
}

ncol_lint <- function(z)
{

  if(is.null(z)) return(0)
  z_num <- NCOL(z[,!sapply(z,is.factor),drop=F])
  facs <- sapply(z,is.factor)
  if(length(facs)>0) z_fac <- sapply(z[,facs,drop=F], nlevels) else
    z_fac <- 0
  if(length(z_fac)==0) z_fac <- 0 else z_fac <- z_fac-1
  return(sum(c(z_num, z_fac)))

}

names_lint <- function(z)
{
  
  unlist(sapply(1:length(z), function(i) 
    if(is.numeric(z[,i])) names(z)[i] else
      paste0(names(z)[i],".",levels(z[,i])[-1])
    ))
  
}

unlist_order_preserving <- function(x)
{

  x_islist <- sapply(x, is.list)
  if(any(x_islist)){

    for(w in which(x_islist)){

      beginning <- if(w>1) x[1:(w-1)] else list()
      end <- if(w<length(x))
        x[(w+1):length(x)] else list()

      is_data_frame <- is.data.frame(x[[w]])
      if(is_data_frame) dfxw <- as.matrix(x[[w]])
      len_bigger_one <- !is_data_frame & length(x[[w]])>1 & is.list(x[[w]])
      if(is_data_frame) x <- append(beginning, list(dfxw)) else
        x <- append(beginning, x[[w]])
      x <- append(x, end)
      if(len_bigger_one) return(unlist_order_preserving(x))

    }

  }

  return(x)

}

get_family_name <- function(dist) gsub(".*(^|/)(.*)/$", "\\2", dist$name)

train_together_ind <- function(train_together)
{

  if(is.list(train_together) & length(train_together )==0) return(NULL)
  nulls <- sapply(train_together, is.null)
  nets <- unique(train_together[!nulls])
  apply(sapply(nets, function(nn)
    sapply(train_together,
           function(tt) if(is.null(tt)) FALSE else nn==tt)), 1, which)


}

sum_cols_smooth <- function(x)
{

  byt <- grepl("by", names(x))
  if(length(byt)==0) return(sum(sapply(x, function(y) NCOL(y$X))))
  # if(sum(byt)==0 & length(x)==1) return(NCOL(x[[1]][[1]]$X))
  if(sum(byt)==0) return(sum(sapply(x, function(y) NCOL(y[[1]]$X))))
  if(sum(byt)==length(byt)) return(sum(sapply(x, sum_cols_smooth)))
  return(sum(sapply(x[byt], sum_cols_smooth)) +
           sum(sapply(x[!byt], function(y) NCOL(y[[1]]$X))))

}

remove_attr <- function(x)
{
  attributes(x) <- NULL
  return(x)
}


get_X_from_linear <- function(lint, newdata = NULL)
{
  
  if(is.null(newdata)){
    if(any(sapply(lint,is.factor))){
      ret <- model.matrix(~ 1 + ., data = lint)[,-1]
    }else{
      ret <- model.matrix(~ 0 + ., data = lint)
    }
  }else{
    ret <- get_X_lin_newdata(linname = names(lint), newdata)
  }
  return(ret)
}

get_X_lin_newdata <- function(linname, newdata)
{
  
  if("(Intercept)" %in% linname)
    newdata$`(Intercept)` <- rep(1, nROW(newdata))
  if("X.Intercept." %in% linname)
    linname[which("X.Intercept." %in% linname)] <- "(Intercept)"
  #if(any(sapply(lint,is.factor))){
    ret <- model.matrix(~ 1 + ., data = newdata[linname])[,-1]
  #}else{
  #  ret <- model.matrix(~ 0 + ., data = newdata[linname])
  #}
  
  return(ret)
  
}

get_names_pfc <- function(pfc) sapply(pfc, "[[", "term")

#### used for the weight history
coefkeras <- function(model)
{
  
  layer_names <- sapply(model$layers, "[[", "name")
  layers_names_structured <- layer_names[
    grep("structured_", layer_names)
  ]
  unlist(sapply(layers_names_structured,
                function(name) model$get_layer(name)$get_weights()[[1]]))
}

#### used in fit.deepregression
WeightHistory <- R6::R6Class("WeightHistory",
                             inherit = KerasCallback,
                             
                             public = list(
                               
                               weights_last_layer = NULL,
                               
                               on_epoch_end = function(batch, logs = list()) {
                                 self$weights_last_layer <-
                                   cbind(self$weights_last_layer,
                                         coefkeras(self$model))
                               }
                             ))




#' Function to subset parsed formulas
#' 
#' @param pfc list of parsed formulas
#' @param type either NULL (all types of coefficients are returned),
#' "linear" for linear coefficients or "smooth" for coefficients of 
#' 
#' @export
get_type_pfc <- function(pfc, type = NULL)
{
  
  linear <- sapply(pfc, function(x) is.null(x$partial_effect) & !is.null(x$coef) & 
                     !(!x$left_from_oz & !is.null(x$right_from_oz)))
  smooth <- sapply(pfc, function(x) !is.null(x$partial_effect) & !is.null(x$coef) & 
                     !(!x$left_from_oz & !is.null(x$right_from_oz)))
  
  if(is.null(type)) type <- c("linear", "smooth") else 
    stopifnot(type %in% c("linear", "smooth"))
  to_return <- linear * ("linear" %in% type) + smooth * ("smooth" %in% type)
  
  return(to_return)
  
}

combine_penalties <- function(penalties, dims)
{
  
  types <- na.omit(sapply(penalties, function(p) if(is.null(p)) return(NA) else p$type))
  output_dims <- na.omit(sapply(penalties, function(p) if(is.null(p)) return(NA) else p$dim))
  null_pen <- sapply(penalties, is.null)
  
  if(any(output_dims>1))
  {
    
    stop("Combined penalties for multi-output not implemented yet.")
    
  }else if(length(penalties)>2){
   
    stop("Combination of more than two penalties not implemented yet.")
     
  }else if(any(types=="l1")){
    
    stop("Combination with Lasso-Penalty not implemented yet.")
    
  }else{
    
    if(all(null_pen)) return(NULL)
    if(any(null_pen)){
      # no penalization in one direction
      
      existing_pen <- penalties[[which(!null_pen)]]
      
      if(existing_pen$type=="l2"){
        
        return(tf$keras$regularizers$l2(existing_pen$values))
      
      }else if(existing_pen$type=="spline"){
        
        return(squaredPenalty(P = kronecker(diag(rep(1, dims[[which(null_pen)]])),
                                        existing_pen$values), 1))
        
      }else{
        
        stop("Not implemented yet.")
        
      }
      
    }else{
      # both directions with penalties
      if(any(types=="spline")){
        
        P1 <- penalties[[1]]$values
        P2 <- penalties[[2]]$values
        
        return(squaredPenalty(
          P = kronecker(P1, diag(rep(1, dims[2]))) + 
            kronecker(diag(rep(1, dims[1])), P2),
          strength = 1
        ))
        
      }else if(all(types=="l2")){
        
        return(tf$keras$regularizers$l2(P1 + P2))
        
      }else{
        
        stop("Not implemented yet.")
        
      }
      
    }
    
  }
}

is_len_zero <- function(x) length(x)==0