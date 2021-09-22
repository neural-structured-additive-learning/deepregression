# function that extracts variables from special symbols in formulas
extract_from_special <- function(x)
{
  if(length(x)>1) return(sapply(x, extract_from_special))
  # remove c()
  if(grepl("c\\(",x))
  {
    x <- gsub("c\\([0-9]+ *, *[0-9]+\\)","", x)
  }
  #
  trimws(
    strsplit(regmatches(x,
                        gregexpr("(?<=\\().*?(?=\\))", x, perl=T))[[1]],
             split = ",")[[1]]
  )
}


remove_brackets <- function(x)
{
  
  if(grepl("^\\(", x))
    return(gsub("^\\(","",gsub("\\)$","",x))) else return(x)
  
}

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

remove_intercept <- function(form) update(form, ~ 0 + . )

frm_to_text <- function(form) Reduce(paste, deparse(form))

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


convertfun_tf <- function(x) tf$constant(x, dtype="float32")

mismatch_brackets <- function(x, logical=TRUE)
{
  
  open_matches <- lengths(regmatches(x, gregexpr("\\{", x)))
  close_matches <- lengths(regmatches(x, gregexpr("\\}", x)))
  
  if(logical) return(open_matches!=close_matches) else
    return(c(open_matches, close_matches))
  
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


# used in subnetwork_init
make_valid_layername <- function(string)
{
  
  gsub("[^a-zA-Z0-9/-]+","_",string)
  
}

#### helper functions for processors

makelayername <- function(term, param_nr, truncate = 30)
{
  
  if(class(term)=="formula") term <- form2text(term)
  return(paste0(strtrim(make_valid_layername(term), truncate), "_", param_nr))
  
}

extractvar <- function(term)
{
  
  all.vars(as.formula(paste0("~", term)))
  
}

#' Extract value in term name
#' 
#' @param term character representing a formula term
#' @param name character; the value to extract
#' @param null_for_missing logical; if TRUE, returns NULL if argument is missing
#' @return the value used for \code{name}
#' @export
#' @examples 
#' extractval("s(a, la = 2)", "la")
#' 
extractval <- function(term, name, null_for_missing = FALSE)
{
  
  if(is.character(term)) term <- as.formula(paste0("~", term))
  inputs <- as.list(as.list(term)[[2]])[-1]
  if(name %in% names(inputs)) return(inputs[[name]])
  if(null_for_missing) return(NULL)
  warning("Argument ", name, " not found. Setting it to some default.")
  if(name=="df") return(NULL) else if(name=="la") return(0.1) else return(NULL)
  
}

extractlen <- function(term, data)
{
  
  vars <- extractvar(term)
  if(is.list(data) & length(vars)==1) return(extractdim(data[[vars]]))
  return(sum(sapply(vars, function(v) NCOL(data[v]))))
  
}

extractdim <- function(x)
{
  
  if(is.null(dim(x))) return(1L)
  return(dim(x)[-1])
  
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


#' Function to index tensors columns
#' 
#' @param A tensor
#' @param start first index
#' @param end last index (equals start index if NULL)
#' @return sliced tensor
#' @export
#' 
tf_stride_cols <- function(A, start, end=NULL)
{
  
  if(is.null(end)) end <- start
  return(
    #tf$strided_slice(A, c(0L,as.integer(start-1)), c(tf$shape(A)[1], as.integer(end)))
    tf$keras$layers$Lambda(function(x) x[,as.integer(start):as.integer(end)])(A)
  )
  
  
}