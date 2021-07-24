
orthog_structured <- function(S,L)
{
  qrL <- qr(L)
  Q <- qr.Q(qrL)
  X_XtXinv_Xt <- tcrossprod(Q)
  Sorth <- S - X_XtXinv_Xt%*%S
  return(Sorth)
}

orthog_structured_smooths <- function(S,P,L)
{

  C <- t(S) %*% L
  qr_C <- qr(C)
  if( any(class(qr_C) == "sparseQR") ){
    rank_C <- qr_C@Dim[2]
  }else{
    rank_C <- qr_C$rank
  }
  Q <- qr.Q(qr_C, complete=TRUE)
  Z <- Q[  , (rank_C + 1) : ncol(Q) ]
  if(is.null(P)) return(S %*% Z) else
    return(list(Snew = S %*% Z,
                Pnew = lapply(P, function(p) t(Z) %*% p %*% Z))
    )

}

orthog_structured_smooths_Z <- function(S,L)
{
  
  C <- t(S) %*% L
  qr_C <- qr(C)
  if( any(class(qr_C) == "sparseQR") ){
    rank_C <- qr_C@Dim[2]
  }else{
    rank_C <- qr_C$rank
  }
  Q <- qr.Q(qr_C, complete=TRUE)
  Z <- Q[  , (rank_C + 1) : ncol(Q) ]
  return(Z)
  
}

orthog_P <- function(P,Z)
{
  return(crossprod(Z,P) %*% Z)
}

orthog_smooth <- function(pcf, zero_cons = TRUE){

  nml <- attr(pcf$linterms, "names")
  nms <- attr(pcf$smoothterms, "names")
  re <- sapply(pcf$smoothterms, function(x) attr(x[[1]], "class")[[1]])=="random.effect"
  # if(!is.null(nms) && grepl(",by_",nms)){
  #   warning("Orthogonalization for s-terms with by-Term currently not supported.")
  # }
  for(nm in nms){

    L <- NULL
    
    Lcontent <- c()

    if(#"(Intercept)" %in% nml &
      !grepl("by", nm) &
      !re[nm] &
      zero_cons)
    {

      L <- matrix(rep(1,NROW(pcf$smoothterms[[nm]][[1]]$X)), ncol=1)
      Lcontent <- c("int")

    }

    if(nm %in% nml & !re[nm]){

      if(!is.null(L))
        L <- cbind(L, pcf$linterms[,nm]) else
          L <- pcf$linterms[,nm]
        Lcontent <- c(Lcontent, "lin")

    }


    if(!is.null(L)){

      X_and_P <- orthog_structured_smooths(
        pcf$smoothterms[[nm]][[1]]$X,
        pcf$smoothterms[[nm]][[1]]$S,
        L
      )

      pcf$smoothterms[[nm]][[1]]$X <- X_and_P[[1]]
      pcf$smoothterms[[nm]][[1]]$S <- X_and_P[[2]]
      pcf$smoothterms[[nm]][[1]]$Lcontent <- Lcontent

    }else{

      pcf$smoothterms[[nm]][[1]]$Lcontent <- Lcontent

    }

  }

  return(pcf)
}



make_orthog <- function(
  pcf,
  retcol = FALSE,
  returnX = FALSE,
  newdata = NULL,
  otherdata = NULL
)
{

  if(is.null(pcf$deepterms)) return(NULL)
  if(is.null(newdata)) n_obs <- nROW(pcf) else n_obs <- nROW(newdata)
  if(n_obs==0){
    if(!is.null(pcf$smoothterms))
      n_obs <- NROW(pcf$smoothterms[[1]][[1]]$X) else
        n_obs <- nROW(pcf$deepterms[[1]])
  }
  nms <- lapply(pcf[c("linterms","smoothterms")], function(x)attr(x,"names"))
  nmsd <- lapply(pcf$deepterms, function(x) attr(x,"names"))
  manoz <- lapply(pcf$deepterms, function(x) attr(x,"manoz"))
  if(!is.null(nms$smoothterms))
    struct_nms <- c(nms$linterms, #unlist(strsplit(nms$smoothterms,",")),
                    nms$smoothterms) else
                      struct_nms <- nms$linterms
  if(is.null(pcf$linterms) & is.null(pcf$smoothterms) & is.null(manoz))
    return(NULL)

  if(is.list(newdata)) newdata <- as.data.frame(newdata)
  
  qList <- lapply(1:length(nmsd), function(i){

    nn <- nmsd[[i]]

    # number of columns removed due to collinearity
    rem_cols <- 0
    # if there is any smooth
    if(length(intersect(nn, struct_nms))>0) 
      X <- matrix(rep(1,n_obs), ncol=1) else
        X <- matrix(ncol=0, nrow=n_obs)
    # Ps <- list()
    # lambdas <- c()
    if(length(intersect(nn, struct_nms)) > 0 | !is.null(manoz[[i]])){

      if(!is.null(manoz[[i]])){ 
        
        X <- cbind(X, do.call("cbind", lapply(manoz[[i]], 
                                              get_X_manoz,
                                              lint = pcf$linterms,
                                              newdata = c(newdata, otherdata))))
        
      }

      for(nm in nn){

        # FIXME: deal with factor variables
        if(nm %in% nms$linterms){ 
          if(!is.null(newdata)){
            X <- cbind(X,newdata[,nm,drop=FALSE])
          }else{
            X <- cbind(X,pcf$linterms[,nm,drop=FALSE])
          }
        }
        if(nm %in% nms$smoothterms){
          
          this_smooth <- pcf$smoothterms[[
            grep(paste0("\\b",nm,"\\b"),nms$smoothterms)
            ]]
          
          if(is.null(newdata)){ 
            if(length(this_smooth)==1)
              this_sX <- this_smooth[[1]]$X else
                this_sX <- this_smooth$X
          }else{
            this_sX <- get_X_from_smooth(this_smooth, newdata)
          }
          
          if(is.null(newdata)){ 
            
            Z_nr <- drop_constant(this_sX)
            X <- cbind(X,Z_nr[[1]])
            rem_cols <- rem_cols + Z_nr[[2]]
            
          }else{
            
            X <- cbind(X, this_sX)
            
          }

        }
      }

      # check for TP
      if(any(length(nn)>1 &  grepl(",", nms$smoothterms))){

        tps_index <- grep(",", nms$smoothterms)
        for(tpi in tps_index){

          if(length(setdiff(unlist(strsplit(nms$smoothterms[tpi],",")), nn))==0){

            this_smooth <- pcf$smoothterms[[tpi]]
            
            if(is.null(newdata)){ 
              if(length(this_smooth)==1)
                this_sX <- this_smooth[[1]]$X else
                  this_sX <- this_smooth$X
            }else{
              this_sX <- get_X_from_smooth(this_smooth, newdata)
            }
            
            X <- cbind(X, this_sX)

          }
        }
      }

    }else{
      return(NULL)
    }

    if(returnX){
      if(retcol) return(NCOL(X)) else
        return(as.matrix(X))
    }

    qrX <- qr(X)
    # if(qrX$rank<ncol(X)){
    #   warning("Collinear features in X")
    #   # qrX <- qr(qrX$qr[,1:qrX$rank])
    # }

    Q <- qr.Q(qrX)
    # coefmat <- tcrossprod(Q)
    if(retcol) return(NCOL(Q)) else
      return(Q)

  })

  return(qList)

}

# for P-Splines
# Section 2.3. of Fahrmeir et al. (2004, Stat Sinica)
centerxk <- function(X,K) tcrossprod(X, K) %*% solve(tcrossprod(K))

orthog <- function(Y, Q)
{

  X_XtXinv_Xt <- tf$linalg$matmul(Q,tf$linalg$matrix_transpose(Q))
  Yorth <- Y - tf$linalg$matmul(X_XtXinv_Xt, Y)
  return(Yorth)

}

orthog_tf <- function(Y, X)
{
  
  Q = tf$linalg$qr(X, full_matrices=TRUE, name="QR")$q
  X_XtXinv_Xt <- tf$linalg$matmul(Q,tf$linalg$matrix_transpose(Q))
  Yorth <- tf$subtract(Y, tf$linalg$matmul(X_XtXinv_Xt, Y))
  
}

orthog_nt <- function(Y,X) Y <- X%*%solve(crossprod(X))%*%crossprod(X,Y)

split_model <- function(model, where = -1)
{

  fun_as_string <- Reduce(paste, deparse(body(model)))
  split_fun <- strsplit(fun_as_string, "%>%")[[1]]
  length_model <- length(split_fun) - 1

  if(where < 0) where <- length_model + where
  # as input is also part of split_fun
  where <- where + 1

  # define functions as strings
  first_part <- paste(split_fun[1:where], collapse = "%>%")
  second_part <- paste(split_fun[c(1,(where+1):(length_model+1))], collapse = "%>%")

  # add missing brackets
  if(mismatch_brackets(first_part))
    first_part <- paste0(first_part, "}")
  if(mismatch_brackets(second_part))
    first_part <- paste0("{", second_part)
  
  # define functions with strings
  first_part <- eval(parse(text = paste0('function(x) ', first_part)))
  second_part <- eval(parse(text = paste0('function(x) ', second_part)))

  return(list(first_part, second_part))

}

### R6 class, not used atm

if(FALSE){

  Orthogonalizer <- R6::R6Class("Orthogonalizer",

                                lock_objects = FALSE,
                                inherit = KerasLayer,

                                public = list(

                                  output_dim = NULL,

                                  kernel = NULL,

                                  initialize = function(inputs) {

                                    self$inputs <- inputs

                                  },

                                  call = function(inputs, training=NULL) {
                                    if(is.null(training))
                                      return(inputs[[1]]) else
                                        return(orthog(inputs[[1]],inputs[[2]]))
                                  }
                                )
  )

  layer_orthog <- function(inputs, ...) {
    create_layer(layer_class = Orthogonalizer,
                 args = list(inputs = inputs)
    )
  }


}

combine_model_parts <- function(deep, deep_top, struct, ox, orthog_fun, shared)
{

  if(is.null(deep) || length(deep)==0){

    if(is.null(shared)) return(struct) else
      return(
        layer_add(list(shared,struct))
      )


  }else if((is.null(struct) || (is.list(struct) && length(struct)==0)) & (is.null(ox) | is.null(ox[[1]]))){

    if(length(deep)==1){

      if(is.null(shared))
        return(deep_top[[1]](deep[[1]])) else
          return(
            layer_add(list(shared,deep_top[[1]](deep[[1]])))
          )

    } # else

    if(is.null(shared))
      return(
        layer_add(
          lapply(1:length(deep), function(j) deep_top[[j]](deep[[j]])))) else
            return(
              layer_add(list(shared,
                             layer_add(lapply(1:length(deep),
                                              function(j) deep_top[[j]](deep[[j]])))))
            )

  }else{

    if(is.null(ox) || length(ox)==0 || (length(ox)==1 & is.null(ox[[1]]))){

      if(is.null(shared))
        return(
          layer_add( append(lapply(1:length(deep),
                                   function(j) deep_top[[j]](deep[[j]])),
                            list(struct))
          )
        ) else
          return(
            layer_add( append(lapply(1:length(deep),
                                     function(j) deep_top[[j]](deep[[j]])),
                              list(struct), list(shared))
            )
          )

    }else{

      if(length(deep) > 1)
        warning("Applying orthogonalization for more than ",
                "one deep model in each predictor.")

      if(is.null(struct) || (is.list(struct) && length(struct)==0)){
        
        if(length(deep)==1){
          return(
              deep_top[[1]](orthog_fun(deep[[1]], ox[[1]]))
              )
        }else{
          
          return(
            layer_add( lapply(1:length(deep),
                              function(j){
                                
                                if(is.null(ox[[j]]))
                                  return(deep_top[[j]](deep[[j]])) else
                                    deep_top[[j]](orthog_fun(deep[[j]],
                                                             ox[[j]]))
                              })) 
          )
          
        }
        
      }
      
      return(
        layer_add( append(lapply(1:length(deep),
                                 function(j){
                                   
                                   if(is.null(ox[[j]]))
                                     return(deep_top[[j]](deep[[j]])) else
                                       deep_top[[j]](orthog_fun(deep[[j]],
                                                                ox[[j]]))
                                 }),
                          struct))
      )
    }
  }
}

drop_constant <- function(X){

  this_notok <- apply(X, 2, function(x) var(x, na.rm=TRUE)==0)
  return(list(X[,!this_notok],
              sum(this_notok))
  )
}

get_X_manoz <- function(m, lint, newdata=NULL)
{
  
  if(is.list(m))
  {
    if(!is.null(newdata)){
      return(get_X_from_smooth(m, newdata))
    }else{
      if(length(m)==1) X <- m[[1]]$X else
        X <- do.call("cbind", lapply(m,"[[","X"))
      return(X)
    }
  }else{
    if(!is.null(newdata)){
      return(get_X_lin_newdata(m, newdata))
    }else{
      return(get_X_lin_newdata(m, lint))
    }
  }
  
}
