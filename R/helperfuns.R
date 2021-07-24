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

#' Function to define orthogonalization connections in the formula
#' 
#' @param form a formula for one distribution parameter
#' @return Returns a list of formula components with ids and 
#' assignments for orthogonalization
#' 
#' 
separate_define_relation <- function(form, specials, specials_to_oz, automatic_oz_check = TRUE)
{
  
  tf <- terms.formula(form, specials = specials)
  has_intercept <- attr(tf, "intercept")
  trmstrings <- attr(tf, "term.labels")
  if(length(trmstrings)==0 & has_intercept)
    return(
      list(list(
        term = "(Intercept)",
        nr = 1,
        left_from_oz = TRUE,
        right_from_oz = NULL
      ))
    )
  variables_per_trmstring <- sapply(trmstrings, function(x) all.vars(as.formula(paste0("~",x))))
  manual_oz <- grepl("%OZ%", trmstrings)
  # do a check for automatic OZ if defined
  # and add it via the %OZ% operator
  if(automatic_oz_check){
    # if no specials_to_oz are present, return function call without automatic oz
    if(is.null(specials_to_oz) | length(specials_to_oz)==0)
      return(separate_define_relation(form, specials = specials, 
                                      specials_to_oz = specials_to_oz,
                                      automatic_oz_check = FALSE))
    # otherwise start checking
    oz_to_add <- rep(list(NULL), length(trmstrings))
    for(i in 1:length(trmstrings)){
      
      if(any(sapply(specials_to_oz, function(nn) grepl(paste0(nn,"\\(.*\\)"), trmstrings[i]))) & 
         !manual_oz[i]){
        # term is checked for automatic orthogonalization
        # find structured term with same variable
        these_vars <- variables_per_trmstring[[i]]
        these_terms <- trmstrings[sapply(1:length(trmstrings), function(j){ 
          !manual_oz[j] & 
            any(sapply(these_vars, function(tv) tv%in%variables_per_trmstring[[j]])) & 
            i != j
        })]
        # TODO: check if this is actually necessary
        if(has_intercept) these_terms <- c("1", these_terms)
        if(length(these_terms)>0) oz_to_add[[i]] <- 
          paste0(" %OZ% (", paste(these_terms, collapse = "+"), ")")
        
      }
    }
    no_changes <- sapply(oz_to_add,is.null)
    if(any(!no_changes)){
      trmstrings[!no_changes] <- mapply(function(x,y) paste0(x, y), 
                                        trmstrings[!no_changes],
                                        oz_to_add[!no_changes])
      form <- as.formula(paste0("~ ", paste(trmstrings, collapse = " + ")))
      return(separate_define_relation(form, specials = specials, 
                                      specials_to_oz = specials_to_oz,
                                      automatic_oz_check = FALSE))
    }
  }
  # define which terms are related to which other terms due to OZ
  terms <- strsplit(trmstrings, "%OZ%", fixed=TRUE)
  terms_left <- lapply(terms, function(x) trimws(x[[1]]))
  terms_right <- lapply(terms, function(trm){
    if(length(trm)>1) remove_brackets(trimws(trm[[2]])) else return(NULL)
  }) 
  terms_right <- lapply(terms_right, function(trm)
  {
    if(is.null(trm)) return(NULL)
    return(trimws(strsplit(trm, "+", fixed=TRUE)[[1]]))
  })

  terms <- lapply(1:length(terms_left), function(i) 
    list(term = terms_left[[i]],
         nr = i,
         left_from_oz = TRUE,
         right_from_oz = NULL
    ))
  
  if(has_intercept & length(intersect(c("(Intercept)","1"), sapply(terms, "[[", "term")))==0)
    terms[[length(terms)+1]] <- list(
      term = "1",
      nr = length(terms)+1,
      left_from_oz = TRUE,
      right_from_oz = NULL
    )
      
  add_terms <- list()
  j <- 1
  
  for(i in 1:length(terms_right)){
    
    if(is.null(terms_right[[i]])) next
    for(k in 1:length(terms_right[[i]])){
      
      is_already_left <- is_equal_not_null(terms_right[[i]][[k]], sapply(terms, "[[", "term"))
      is_already_right <- FALSE
      if(length(add_terms)>0)
        is_already_right <- is_equal_not_null(terms_right[[i]][[k]], sapply(add_terms, "[[", "term"))
      if(any(is_already_left)){
        terms[[which(is_already_left)]]$right_from_oz <- 
          c(terms[[which(is_already_left)]]$right_from_oz, i)
      }else if(any(is_already_right)){
        add_terms[[which(is_already_right)]]$right_from_oz <- 
          c(add_terms[[which(is_already_right)]]$right_from_oz, i)
      }else{ # add
        add_terms[[j]] <- list(
          term = terms_right[[i]][[k]],
          nr = length(terms) + j,
          left_from_oz = FALSE,
          right_from_oz = i
        )
        j <- j + 1
      }
      
    }
    
  }
  
  terms <- c(terms, add_terms)
  
  if(has_intercept){
    
    terms[[which(sapply(terms, "[[", "term")=="1")]]$left_from_oz <- TRUE
    
  }
  
  return(terms)
  
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

# #### from mgcv
# uniquecombs <- function(x,ordered=FALSE) {
#   ## takes matrix x and counts up unique rows
#   ## `unique' now does this in R
#   if (is.null(x)) stop("x is null")
#   if (is.null(nrow(x))||is.null(ncol(x))) x <- data.frame(x)
#   recheck <- FALSE
#   if (inherits(x,"data.frame")) {
#     xoo <- xo <- x
#     ## reset character, logical and factor to numeric, to guarantee that text versions of labels
#     ## are unique iff rows are unique (otherwise labels containing "*" could in principle
#     ## fool it).
#     is.char <- rep(FALSE,length(x))
#     for (i in 1:length(x)) {
#       if (is.character(xo[[i]])) {
#         is.char[i] <- TRUE
#         xo[[i]] <- as.factor(xo[[i]])
#       }
#       if (is.factor(xo[[i]])||is.logical(xo[[i]])) x[[i]] <- as.numeric(xo[[i]])
#       if (!is.numeric(x[[i]])) recheck <- TRUE ## input contains unknown type cols
#     }
#     #x <- data.matrix(xo) ## ensure all data are numeric
#   } else xo <- NULL
#   if (ncol(x)==1) { ## faster to use R
#     xu <- if (ordered) sort(unique(x[,1])) else unique(x[,1])
#     ind <- match(x[,1],xu)
#     if (is.null(xo)) x <- matrix(xu,ncol=1,nrow=length(xu)) else {
#       x <-  data.frame(xu)
#       names(x) <- names(xo)
#     }
#   } else { ## no R equivalent that directly yields indices
#     if (ordered) {
#       chloc <- Sys.getlocale("LC_CTYPE")
#       Sys.setlocale("LC_CTYPE","C")
#     }
#     ## txt <- paste("paste0(",paste("x[,",1:ncol(x),"]",sep="",collapse=","),")",sep="")
#     ## ... this can produce duplicate labels e.g. x[,1] = c(1,11), x[,2] = c(12,2)...
#     ## solution is to insert separator not present in representation of a number (any
#     ## factor codes are already converted to numeric by data.matrix call above.)
#     txt <- paste("paste0(",paste("x[,",1:ncol(x),"]",sep="",collapse=",\"*\","),")",sep="")
#     xt <- eval(parse(text=txt)) ## text representation of rows
#     dup <- duplicated(xt)       ## identify duplicates
#     xtu <- xt[!dup]             ## unique text rows
#     x <- x[!dup,]               ## unique rows in original format
#     #ordered <- FALSE
#     if (ordered) { ## return unique in same order regardless of entry order
#       ## ordering of character based labels is locale dependent
#       ## so that e.g. running the same code interactively and via
#       ## R CMD check can give different answers.
#       coloc <- Sys.getlocale("LC_COLLATE")
#       Sys.setlocale("LC_COLLATE","C")
#       ii <- order(xtu)
#       Sys.setlocale("LC_COLLATE",coloc)
#       Sys.setlocale("LC_CTYPE",chloc)
#       xtu <- xtu[ii]
#       x <- x[ii,]
#     }
#     ind <- match(xt,xtu)   ## index each row to the unique duplicate deleted set
#
#   }
#   if (!is.null(xo)) { ## original was a data.frame
#     x <- as.data.frame(x)
#     names(x) <- names(xo)
#     for (i in 1:ncol(xo)) {
#       if (is.factor(xo[,i])) { ## may need to reset factors to factors
#         xoi <- levels(xo[,i])
#         x[,i] <- if (is.ordered(xo[,i])) ordered(x[,i],levels=1:length(xoi),labels=xoi) else
#           factor(x[,i],levels=1:length(xoi),labels=xoi)
#         contrasts(x[,i]) <- contrasts(xo[,i])
#       }
#       if (is.char[i]) x[,i] <- as.character(x[,i])
#       if (is.logical(xo[,i])) x[,i] <- as.logical(x[,i])
#     }
#   }
#   if (recheck) {
#     if (all.equal(xoo,x[ind,],check.attributes=FALSE)!=TRUE)
# warning("uniquecombs has not worked properly")
#   }
#   attr(x,"index") <- ind
#   x
# } ## uniquecombs
#
# ### from mgcv
# compress_data <- function(dat, m = NULL)
# {
#   d <- ncol(dat) ## number of variables to deal with
#   n <- nrow(dat) ## number of data/cases
#   if (is.null(m)) m <- if (d==1) 1000 else if (d==2) 100 else 25 else
#     if (d>1) m <- round(m^{1/d}) + 1
#
#   mf <- mm <- 1 ## total grid points for factor and metric
#   for (i in 1:d) if (is.factor(dat[,i])) {
#     mf <- mf * length(unique(as.vector(dat[,i])))
#   } else {
#     mm <- mm * m
#   }
#   if (is.matrix(dat[[1]])) { ## must replace matrix terms with vec(dat[[i]])
#     dat0 <- data.frame(as.vector(dat[[1]]))
#     if (d>1) for (i in 2:d) dat0[[i]] <- as.vector(dat[[i]])
#     names(dat0) <- names(dat)
#     dat <- dat0;rm(dat0)
#   }
#   xu <- uniquecombs(dat,TRUE)
#   if (nrow(xu)>mm*mf) { ## too many unique rows to use only unique
#     for (i in 1:d) if (!is.factor(dat[,i])) { ## round the metric variables
#       xl <- range(dat[,i])
#       xu <- seq(xl[1],xl[2],length=m)
#       dx <- xu[2]-xu[1]
#       kx <- round((dat[,i]-xl[1])/dx)+1
#       dat[,i] <- xu[kx] ## rounding the metric variables
#     }
#     xu <- uniquecombs(dat,TRUE)
#   }
#   k <- attr(xu,"index")
#   ## shuffle rows in order to avoid induced dependencies between discretized
#   ## covariates (which can mess up gam.side)...
#   ## Any RNG setting should be done in routine calling this one!!
#
#   ii <- sample(1:nrow(xu),nrow(xu),replace=FALSE) ## shuffling index
#
#   xu[ii,] <- xu  ## shuffle rows of xu
#   k <- ii[k]     ## correct k index accordingly
#   ## ... finished shuffle
#   ## if arguments were matrices, then return matrix index
#   if (length(k)>n) k <- matrix(k,nrow=n)
#   k -> attr(xu,"index")
#   xu
# }

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

# TODO: ADD fac, lasso ridge fm offset
# get contents from formula
get_contents <- function(lf, data, df,
                         network_names,
                         intercept = TRUE,
                         defaultSmoothing = NULL,
                         absorb_cons = TRUE,
                         null_space_penalty = FALSE,
                         hat1 = TRUE,
                         sp_scale = 1, 
                         anisotropic = TRUE,
                         image_var = list(),
                         nr_param,
                         zero_constraint_for_smooths = TRUE,
                         variational_options = list())
  {
  # extract which parts are modelled as deep parts
  # which by smooths, which linear
  if(is.character(lf)) lf <- as.formula(lf)
  
  # get variable_names
  variable_names <- names(data)
  if(is.null(variable_names) | any(variable_names==""))
    stop("If data is a list, names must be given.")
  
  specials <- c("s", "te", "ti", "vc", "vvc", "fac", "lasso", "ridge", "fm", "offset", "vi", network_names)
  overlap <- setdiff(specials[-length(specials)], netnames)
  
  if(length(overlap)>0)
    stop("Please rename networks in the list of deep models with name", paste(overlap, collapse=", "), ".")
  
  tf <- terms.formula(lf, specials=specials, data=data)

  if(length(attr(tf, "term.labels"))==0){
    if(intercept & attr(tf,"intercept")){
      if(is.data.frame(data)) linterms <- data.frame(a=rep(1,nrow(data))) else
        linterms <- data.frame(a=rep(1,nROW(data)))
      names(linterms) <- "(Intercept)"
      attr(linterms, "names") <- names(linterms)
      ret <- list(linterms = linterms,
                  smoothterms = NULL,
                  deepterms = NULL)
      attributes(ret) <-
        c(attributes(ret),
          list(formula = lf,
               df = df,
               variable_names = variable_names,
               network_names = network_names,
               intercept = intercept,
               defaultSmoothing = defaultSmoothing)
        )
      return(ret)
    }else{ return(NULL) }
  }
  trmstrings <- attr(tf, "term.labels")
  # if(length(setdiff(c(gsub("(.*)\\(.*\\)","\\1",trmstrings),
  #                     variable_names),
  #                   specials))>0)
  #   stop("It seems that you are using non-valid terms in the formula ",
  #        "or specified a list_of_deep_models without names.")

  # check for weird line break behaviour produced by terms.formula
  trmstrings <- unname(sapply(trmstrings, function(x)
    gsub("\\\n\\s+", "", x, fixed=F)))
  # check for missing covariates in data
  for(j in trmstrings)
  {
    if(!grepl("\\(",j) | !grepl("\\)",j)){
      if(xor(!grepl("\\(",j),  !grepl("\\)",j))){
        stop("Terms in formula with only one parantheses.")
      }else{
        # make pseudo parantheses so regmatch detects variable
        # in the following lines
        j <- paste0("(",j,")")
      }
    }
    vars <- extract_from_special(j)
    # drop terms that specify a s-term specification
    vars <- vars[!grepl("=", vars, fixed=T)]
    # replace . in formula
    if(length(vars)==1 && vars==".")
    {
      ff <- as.character(lf)[[2]]
      net_w_dot <- sapply(network_names, function(x) grepl(paste0(x,"\\("),j))
      if(grepl("d\\(",j) | any(net_w_dot))
        ff <- gsub("\\.", paste(variable_names, collapse=","), ff) else
          ff <- gsub(".", paste(variable_names, collapse="+"), ff)
      return(get_contents(lf = as.formula(paste0("~ ", ff)),
                          data = data,
                          df = df,
                          variable_names = variable_names,
                          intercept = intercept,
                          network_names = network_names,
                          defaultSmoothing = defaultSmoothing))
    }
    whatsleft <- setdiff(vars, c(variable_names, "FALSE", "TRUE"))
    if(length(whatsleft) > 0){
      if(grepl(":", whatsleft))
        stop("Linear interactions such as ", whatsleft[1],
             " have to be defined manually at the moment.") else
               stop(paste0("data for ", paste(whatsleft, collapse = ","), " in ",
                           j, " not found"))
    }
  }

  ##################################### linear start #########################################

  #
  terms <- sapply(trmstrings, function(trm) as.call(parse(text=trm))[[1]],
                  simplify=FALSE)
  # get formula environment
  # frmlenv <- environment(formula)
  # get linear terms
  desel <- unlist(attr(tf, "specials"))
  # if(is.data.frame(data)){
  #   if(!is.null(desel)) linterms <-
  #       data[,attr(tf, "term.labels")[-1*desel], drop=FALSE] else
  #       linterms <- data[,attr(tf, "term.labels"), drop=FALSE]
  # }else{
  if(!is.null(desel)){
    ind <- attr(tf, "term.labels")[-1*desel]
    if(length(ind)!=0) linterms <- as.data.frame(data[ind]) else
      linterms <- data.frame(dummy=1:nROW(data))[character(0)]
  }else{
    # else
    #     stop("When using only structured terms, data must be a data.frame")
    if(length(attr(tf,"term.labels"))>0)
      linterms <- as.data.frame(data[attr(tf, "term.labels")]) else
        linterms <- data.frame(dummy=1:nROW(data))[character(0)]
      # }
  }
  if(intercept & attr(tf,"intercept"))#{
    # if(NCOL(linterms)==0)
    if(NROW(linterms)==0)
      linterms <- data.frame("(Intercept)" = rep(1,nROW(data))) else
        linterms <- cbind("(Intercept)" = rep(1,nROW(data)),
                          as.data.frame(linterms))# else

  attr(linterms, "names") <- names(linterms)

  ##################################### linear end #########################################

  ##################################### smooths start #########################################

  # get gam terms
  spec <- attr(tf, "specials")
  sTerms <- terms[sort(unlist(spec[names(spec) %in% c("s", "te", "ti")]))]
  # if(any(!sapply(spec[c("te","ti")], is.null)))
  #  warning("2-dimensional smooths and higher currently not well tested.")
  if(length(sTerms)>0)
  {
    names_sTerms <- names(sTerms)
    terms_w_s <- lapply(names(sTerms), extract_from_special)
    terms_w_s <- lapply(terms_w_s, function(x) sapply(x, function(y){
      if(grepl("by.*\\=",y)) return(trimws(gsub("by.*\\=(.*)","\\1",y))) else return(y)}))
    terms_w_s <- lapply(terms_w_s, function(x) x[!grepl("=", x, fixed=T)])
    smoothterms <-
      lapply(sTerms,
             function(t) {
             smoothCon(eval(t),
                         data=data.frame(data[setdiff(unname(unlist(terms_w_s)), 
                                                      c("TRUE", "FALSE"))]),
                         knots=NULL, absorb.cons = absorb_cons,
                         null.space.penalty = null_space_penalty)
      })

    # ranks <- sapply(smoothterms, function(x) rankMatrix(x$X, method = 'qr',
    # warn.t = FALSE))
    if(is.null(df)) df <- pmax(min(sapply(smoothterms,
                                          function(x){ 
                                            
                                            if(length(x)>1 & x[[1]]$by=="NA") 
                                              return(sum(sapply(x, "[[", "df")))
                                            if(x[[1]]$by!="NA") return(min(sapply(x,"[[","df")))
                                            return(x[[1]]$df)
                               
                               }) - null_space_penalty), 1)
    # check correct length when df is a vector
    smooths_w_pen <- sapply(smoothterms,function(x) is.null(x[[1]]$sp))
    if(length(df)>1)
      stopifnot(sum(smooths_w_pen)==length(df)) else if(length(smoothterms)>1 & 
                                                        sum(smooths_w_pen)>0)
        df <- as.list(rep(df, sum(smooths_w_pen)))

    if(!is.list(df))
    {
      message("Converting vector of df values to list.")
      df <- as.list(df)
    }
    
    if(is.null(defaultSmoothing))
      defaultSmoothing = function(st, this_df){
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
              )["lambda"]/sp_scale + 
                null_space_penalty
            })
            return(st)
          }else{
            X <- do.call("cbind", lapply(st,"[[","X"))
          }
        }
        st[[1]]$sp = DRO(X, df = this_df, dmat = S, hat1 = hat1)["lambda"]/sp_scale + 
          null_space_penalty
        return(st)
      }
    if(sum(smooths_w_pen)>0)
      smoothterms[smooths_w_pen] <-
      lapply(1:sum(smooths_w_pen),
             function(i)
               defaultSmoothing(
                 smoothterms[smooths_w_pen][[i]],
                 df[[i]]
               )
      )
    attr(smoothterms, "names") <-
      unlist(lapply(names_sTerms,
                    function(x){
                      vars <- extract_from_special(x)
                      vars <- vars[!grepl("=", vars, fixed=T) | grepl("by.*\\=",vars)]
                      # rep <- FALSE
                      # if(any(grepl("by.*\\=",vars))){
                      #   fac <- trimws(gsub("by.*\\=(.*)","\\1",vars[grepl("by.*\\=",vars)]))
                      #   rep <- TRUE
                      #   }
                      # vars[grepl("by.*\\=",vars)] <-
                      # gsub("(\\s+)\\=(\\s+)","_",vars[grepl("by.*\\=",vars)])
                      #ret <-
                      paste(vars, collapse=",")
                      # if(rep) paste0(ret, 1:nlevels(data[[fac]])) else ret
                    }))
    # values in smooth construct list have the following items
    # (see also ?mgcv::smooth.construct)
    #
    # X: model matrix
    # S: psd penalty matrix
    # rank: array with ranks of penalties
    # null.space.dim: dimension of penalty null space
    # C: identifiability constraints on term (per default sum-to-zero constraint)
    # and potential further entries
  }else{
    smoothterms <- NULL
  }

  ##################################### smooths end #########################################

  ##################################### deep start #########################################

  # get deep terms
  dterms <- sapply(paste0("^",network_names,"\\("), function(x) trmstrings[grepl(x,trmstrings)])
  if(all(sapply(dterms,length)==0)){
    deepterms <- NULL
  }else{
    deepterms <- lapply(dterms[sapply(dterms,length)>0], function(dt){
      if(length(dt) == 1 && grepl("%OZ%",dt)){
       dt_split <- trimws(strsplit(dt, "%OZ%")[[1]])
       dt <- dt_split[[1]]
       dtoz <- dt_split[[2]]
      }else{
        dtoz <- NULL
      }
      ##### the actual deep part

      this_var <- extract_from_special(dt)
      
      if(!(length(this_var)==1 && this_var%in%names(image_var))){
        
        if(is.data.frame(data)){
          deepterms <- data[,this_var,drop=FALSE]
          attr(deepterms, "names") <- names(deepterms)
        }else{
          deepterms <- data[extract_from_special(dt)]
          
          if(length(this_var)>1)
            deepterms <- as.data.frame(deepterms)
          
        }
        attr(deepterms, "names") <- names(deepterms)
        
      }else{
        
        deepterms <- data.frame(matrix(nrow=c(nROW(data), ncol=0)))
        attr(deepterms, "dims") <- c(nrow(deepterms), image_var[[names(image_var)==this_var]])
        
      }
      
      ##### end actual deep part

      if(!is.null(dtoz)){
        dtoz <- trimws(strsplit(gsub("^\\((.*)\\)$","\\1",dtoz),"\\+")[[1]])
        manoz <- lapply(dtoz, function(ddd){
          if(any(sapply(c("s\\(","ti\\(","te\\("), function(t) grepl(t,ddd))))
          {

            terms_w_s <- extract_from_special(ddd)
            terms_w_s <- sapply(terms_w_s, function(y){
              if(grepl("by.*\\=",y)) return(trimws(gsub("by.*\\=(.*)","\\1",y))) else return(y)})
            terms_w_s <- terms_w_s[!grepl("=", terms_w_s, fixed=T)]
            st <- smoothCon(eval(parse(text = ddd)),
                            data=data.frame(data[unname(unlist(terms_w_s))]),
                            knots=NULL, absorb.cons = absorb_cons,
                            null.space.penalty = null_space_penalty)
            # if(length(st)==1) X <- st[[1]]$X else
            #   X <- do.call("cbind", lapply(st,"[[","X"))
            # return(X)
            return(st)
          }else{
            return(ddd)
            # if(is.data.frame(data))
            #   return(model.matrix(~ 1+data[,ddd,drop=TRUE])[,-1]) else
            #     return(model.matrix(~ 1+data[[ddd]])[,-1])
          }
        })
        # manoz <- do.call("cbind", manoz)

        attr(deepterms,"manoz") <- manoz

      }else{
        attr(deepterms,"manoz") <- NULL
      }
      return(deepterms)
    })
    if(length(network_names)==1)
      names(deepterms) <- rep(network_names, length(deepterms)) else
        names(deepterms) <- network_names[sapply(dterms,length)>0]
  }

  ##################################### deep end #########################################

  ##################################### VCs ##############################################
  
  if(!is.null(attr(tf, "specials")$vc) | !is.null(attr(tf, "specials")$vvc)){
       
    if(!is.null(attr(tf, "specials")$vvc))
      stop("Not implemented yet.")
    
    vclist <- lapply(attr(tf, "specials")$vc, function(i) 
      build_vc(terms[[i]], data, name = paste0("tp_layer_", nr_param, "_", i)))
    names(vclist) <- paste0(sapply(terms[attr(tf, "specials")$vc], 
                                   function(x) gsub("\\s|\\)","", gsub("\\(|,|=", "_",  deparse(x)))),
                            "_param_", nr_param)
    
    if(is.null(deepterms)) deepterms <- vclist else
      deepterms <- c(deepterms, vclist)
    
  }
  
  ##################################### VCs end ##########################################
  
  ret <- list(linterms = linterms,
              smoothterms = smoothterms,
              deepterms = deepterms)

  attributes(ret) <-
    c(attributes(ret),
      list(formula = lf,
           df = df,
           variable_names = variable_names,
           network_names = network_names,
           intercept = intercept,
           defaultSmoothing = defaultSmoothing)
    )
  
  # check for zero ncol linterms
  if(NCOL(ret$linterms)==0) ret$linterms <- list(NULL)
  
  # orthognalize smooths
  ret <- orthog_smooth(ret, zero_cons = zero_constraint_for_smooths)
  attr(ret,"zero_cons") <- zero_constraint_for_smooths

  return(ret)
}

get_contents_newdata <- function(pcf, newdata)
  lapply(pcf, function(x) get_contents(lf = attr(x, "formula"),
                                       data = newdata,
                                       df = attr(x, "df"),
                                       variable_names = attr(x, "variable_names"),
                                       network_names = attr(x, "network_names"),
                                       intercept = attr(x, "intercept"),
                                       defaultSmoothing = attr(x, "defaultSmoothing")))

make_cov <- function(pcf,
                     newdata=NULL,
                     pred = !is.null(newdata), 
                     olddata=NULL,
                     orthogonalize = TRUE,
                     ...
                     ){

  if(is.null(newdata)){
    input_cov <- lapply(pcf, function(x){
      if(is.null(x$deepterms)) return(NULL) else
        return(x$deepterms)
    })
  }else{
    input_cov <- lapply(pcf, function(x){
      if(length(intersect(sapply(x$deepterms,
                                 function(y) names(y)),names(newdata)))>0 | 
         any(sapply(x$deepterms,function(x)class(x)[1])=="vcdata")){
        ret <- lapply(x$deepterms, function(y){
          if(is.data.frame(y)){
            return(as.data.frame(newdata[names(y)]))
          }else if("vcdata" %in% class(y)){
            return(newdata_vc(y, newdata))
          }else{
            return(newdata[names(y)])
          }
        })
        
        return(ret)

      }else if(is.list(x$deepterms) & all(sapply(x$deepterms, class)=="data.frame")){

        return(lapply(x$deepterms, function(y) data.frame(newdata[names(y)])))
# 
#       }else if(is.list(x$deepterms) & ){  
        
      }else{ return(NULL) }
    })
  }
  if(is.list(input_cov) & all(sapply(input_cov, is.list)))
    input_cov <- unlist(input_cov, recursive = F, use.names = F)
  input_cov_isdf <- sapply(input_cov, is.data.frame)
  if(sum(input_cov_isdf)>0)
    input_cov[which(input_cov_isdf)] <-
    lapply(input_cov[which(input_cov_isdf)], as.matrix)

  # if(!is.null(newdata) & pred)
  #   pcfnew <- get_contents_newdata(pcf, newdata)

  for(i in 1:length(pcf)){

    x = pcf[[i]]
    ret <- NULL
    if(!is.null(x$linterms))
      ret <- get_X_from_linear(x$linterms, newdata = newdata)
    if(!is.null(x$smoothterms))
    {
      if(!is.null(newdata)){
        Xp <- lapply(x$smoothterms, function(sm) get_X_from_smooth(sm, newdata))
      }else{
        Xp <- lapply(x$smoothterms, function(x)
          do.call("cbind", lapply(x, "[[", "X")))
      }
      st <- do.call("cbind", Xp)
      if(!is.null(ret)){
        ret <- cbind(as.data.frame(ret), st)

      }else{
        ret <- st
      }
      ret <- array(as.matrix(ret),
                   dim = c(nrow(ret),ncol(ret)))
    }

    if(i==2 & !is.null(attr(x,"minval")) & pred)
      ret <- ret - attr(x, "minval")
    input_cov <- c(input_cov, list(ret))

  }

  # just use the ones with are actually modeled
  input_cov <- input_cov[!sapply(input_cov, function(x) is.null(x) |
                                   (length(x)==1 && is.null(x[[1]])) |
                                   NCOL(x)==0)]
  input_cov <- unlist_order_preserving(input_cov)
  list_len_1 <- sapply(input_cov, function(x) is.list(x) & length(x)==1)
  input_cov[list_len_1] <- lapply(input_cov[list_len_1], function(x) x[[1]])
  input_cov[sapply(lapply(input_cov,dim),is.null)] <-
    lapply(input_cov[sapply(lapply(input_cov,dim),is.null)],
           function(x) matrix(x, ncol=1))
  input_cov_isdf <- sapply(input_cov, is.data.frame)
  if(sum(input_cov_isdf)>0)
    input_cov[which(input_cov_isdf)] <-
    lapply(input_cov[which(input_cov_isdf)], as.matrix)
  which_to_convert <- !sapply(input_cov,function(ic){is.factor(ic) | 
      any(class(ic)=="placeholder") | length(dim(ic))>2})
  input_cov[which_to_convert] <- lapply(input_cov[which_to_convert], as.matrix)
  
  ### OZ
  
  # if(!is.null(data) & is.null(index)){
  #   pfc <- get_contents_newdata(pfc, data)
  # if(!is.null(newdata))
  ox <- oxx <- lapply(pcf, make_orthog, 
                      newdata = newdata, 
                      otherdata = olddata,
                      ...) # else
  # ox <- lapply(pcf, make_orthog, newdata = olddata)
  ox <- unlist(lapply(ox, function(x_per_param)
    if(is.null(x_per_param)) return(NULL) else
      (lapply(x_per_param[!sapply(x_per_param,is.null)], function(x)
        as.matrix(x)))), recursive=F)

  input_cov <- 
    append(
      c(unname(input_cov)),unname(ox[!sapply(ox, is.null)]))
  
  if(!is.null(list(...)$returnX))
    attr(input_cov, "ox") <- oxx
  
  ####
  return(input_cov)

}

build_vc <- function(term, data, name){
  
  org_vars <- extract_from_special(list(term))
  org_vars_org <- org_vars
  lambdasind <- grepl("lambda\\s*=", org_vars)
  if(any(lambdasind)){
    lambdas <- gsub(".*(lambda\\s*=\\s*c\\(.*\\))\\)||,.*\\)", "\\1", deparse(term))
    org_vars <- org_vars[!lambdasind]
    eval(parse(text=lambdas))
  }else{
    lambda <- 1
  }
  num <- smoothCon(eval(parse(text = paste0("s(", paste(org_vars[-2], collapse=", "), ")"))),
                   data = as.data.frame(data[org_vars[1]]))
  fac <- data[[org_vars[2]]]
  if(!is.factor(fac)) stop("The by-term of vc terms must be a factor variable.")
  nlev <- nlevels(fac)
  fac <- as.integer(fac)-1
  vcterms <- list(cbind(num[[1]]$X, fac))
  names(vcterms)
  Ptp <- do.call("tp_penalty", c(list(num[[1]]$S[[1]], diag(rep(1,nlev))), as.list(lambda)))
  names(vcterms) <- gsub("\\s|\\)","", gsub("\\(|,|=", "_",  deparse(term)))
  attr(vcterms, "layer") <- vc_block(ncol(num[[1]]$X), nlev, penalty=quadpen(Ptp), 
                                     name=name)
  attr(vcterms, "org_features") <- org_vars_org[1:2]
  attr(vcterms, "smterm") <- num
  class(vcterms) <- c("vcdata", "list")
  return(vcterms)
  
}

newdata_vc <- function(vcobj, newdata)
{
  
  pm <- PredictMat(attr(vcobj, "smterm")[[1]],
                   as.data.frame(newdata[attr(vcobj, "org_features")[1]]))
  return(cbind(pm, as.integer(newdata[[attr(vcobj, "org_features")[2]]])-1))
  
}


get_names <- function(x)
{

  lret <- list(linterms = NULL,
               smoothterms = NULL,
               deepterms = NULL)
  if(!is.null(x$linterms)) lret$linterms <- names_lint(x$linterms)
  if(!is.null(x$smoothterms)) lret$smoothterms <-
      c(sapply(x$smoothterms,function(x) sapply(x, "[[", "label")))
  if(!is.null(x$deepterms)) lret$deepterms <- names(x$deepterms)
  return(lret)
}

get_indices <- function(x)
{
  if(!is.null(x$linterms) &
     !(length(x$linterms)==1 & is.null(x$linterms[[1]])))
    ncollin <- ncol_lint(x$linterms) else ncollin <- 0
    if(!is.null(x$smoothterms))
      bsdims <- unlist(lapply(x$smoothterms, function(y){
        if(is.null(y[[1]]$margin) & y[[1]]$by=="NA")
          return(ncol(y[[1]]$X)) else if(
            is.null(y[[1]]$margin) & y[[1]]$by!="NA")
            return(sapply(y, "[[", "bs.dim")) else{
              # Tensorprod
              if(grepl("ti\\(", y[[1]]$label))
                res <- prod(sapply(y[[1]]$margin,function(sp)ncol(sp$X))) else
                  res <- prod(sapply(y[[1]]$margin,"[[", "bs.dim"))
              # check z2s constraint
              if(!is.null(y[[1]]$X) && NCOL(y[[1]]$X)==res-1)
                return(res-1) else return(res)
            }
      })) else bsdims <- c()
      ind <- if(ncollin > 0) seq(1, ncollin, by = 1) else c()
      end <- if(ncollin > 0) ind else c()
      if(length(bsdims) > 0) ind <- c(ind, max(c(ind,0))+1, max(c(ind+1,1)) +
                                        cumsum(bsdims[-length(bsdims)]))
      if(length(bsdims) > 0) end <- c(end, max(c(end,0)) +
                                        cumsum(bsdims))

      return(data.frame(start=ind, end=end,
                        type=c(rep("lin",ncollin),
                               rep("smooth",length(bsdims))))
      )
}


coefkeras <- function(model)
{

  layer_names <- sapply(model$layers, "[[", "name")
  layers_names_structured <- layer_names[
    grep("structured_", layer_names)
    ]
  unlist(sapply(layers_names_structured,
                function(name) model$get_layer(name)$get_weights()[[1]]))
}

make_cv_list_simple <- function(data_size, folds, seed = 42, shuffle = TRUE)
{

  set.seed(seed)
  suppressWarnings(
    mysplit <- split(sample(1:data_size),
                     f = rep(1:folds, each = data_size/folds))
  )
  lapply(mysplit, function(test_ind)
    list(train_ind = setdiff(1:data_size, test_ind),
         test_ind = test_ind))

}

extract_cv_result <- function(res, name_loss = "loss", name_val_loss = "val_loss"){

  losses <- sapply(res, "[[", "metrics")
  trainloss <- data.frame(losses[name_loss,])
  validloss <- data.frame(losses[name_val_loss,])
  weightshist <- lapply(res, "[[", "weighthistory")

  return(list(trainloss=trainloss,
              validloss=validloss,
              weight=weightshist))

}

#' Plot CV results from deepregression
#'
#' @param x \code{drCV} object returned by \code{cv.deepregression}
#' @param what character indicating what to plot (currently supported 'loss'
#' or 'weights')
#' @param ... further arguments passed to \code{matplot}
#'
#' @export
#'
plot_cv <- function(x, what=c("loss","weight"), ...){

  .pardefault <- par()
  cres <- extract_cv_result(x)

  what <- match.arg(what)

  if(what=="loss"){

    loss <- cres$trainloss
    mean_loss <- apply(loss, 1, mean)
    vloss <- cres$validloss
    mean_vloss <- apply(vloss, 1, mean)

    par(mfrow=c(1,2))
    matplot(loss, type="l", col="black", ..., ylab="loss", xlab="epoch")
    points(1:(nrow(loss)), mean_loss, type="l", col="red", lwd=2)
    abline(v=which.min(mean_loss), lty=2)
    matplot(vloss, type="l", col="black", ...,
            ylab="validation loss", xlab="epoch")
    points(1:(nrow(vloss)), mean_vloss, type="l", col="red", lwd=2)
    abline(v=which.min(mean_vloss), lty=2)
    suppressWarnings(par(.pardefault))

  }else{

    stop("Not implemented yet.")

  }

  invisible(NULL)

}

#' Function to get the stoppting iteration from CV
#' @param res result of cv call
#' @param thisFUN aggregating function applied over folds
#' @param loss which loss to use for decision
#' @param whichFUN which function to use for decision
#'
#' @export
stop_iter_cv_result <- function(res, thisFUN = mean,
                                loss = "validloss",
                                whichFUN = which.min)
{

  whichFUN(apply(extract_cv_result(res)[[loss]], 1, FUN=thisFUN))

}

#' Generate folds for CV out of one hot encoded matrix
#'
#' @param mat matrix with columns corresponding to folds
#' and entries corresponding to a one hot encoding
#' @param val_train the value corresponding to train, per default 0
#' @param val_test the value corresponding to test, per default 1
#'
#' @details
#' \code{val_train} and \code{val_test} can both be a set of value
#'
#' @export
make_folds <- function(mat, val_train=0, val_test=1)
{

  apply(mat, 2, function(x){
    list(train = which(x %in% val_train),
         test = which(x %in% val_test))
  })

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

applySumToZero <- function(X, apply=TRUE)
{
  if(apply)
    return(orthog_structured_smooths(X, NULL, matrix(rep(1,nrow(X)),ncol=1)))
  return(X)
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

get_X_from_smooth <- function(sm, newdata)
{
  
  if(length(sm)==1 & sm[[1]]$by=="NA" & !("random.effect" %in% attr(sm[[1]], "class"))){
    sm <- sm[[1]]
    sterms <- sm$term
    Lcontent <- sm$Lcontent
    pm <- PredictMat(sm,as.data.frame(newdata[sterms]))
    if(length(Lcontent)>0)
    {
      if("int" %in% Lcontent)
        thisL <- matrix(rep(1,NROW(newdata[[1]])), ncol=1)
      if("lin" %in% Lcontent)
        thisL <- cbind(thisL, newdata[[sterms]])
    }else thisL <- NULL
    if(is.null(thisL))
      return(pm) else
        return(
          orthog_structured_smooths(
            S = pm, P = NULL, L = thisL
          )
        )
  }else if("random.effect" %in% attr(sm[[1]], "class")){
    sterms <- sm[[1]]$term
    pm <- PredictMat(sm[[1]],as.data.frame(newdata[sterms]))
    return(pm)
  }else{
    sterms <- c(sm[[1]]$term, sm[[1]]$by)
    do.call("cbind", lapply(sm, function(smm)
      applySumToZero(PredictMat(smm,as.data.frame(newdata[sterms])),
                     apply = FALSE)))
  }
  
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

shape_trainable_weights <- function(mod) sapply(mod$model$trainable_weights, 
                                                function(x) c(as.matrix(tf$shape(x))))

uniqueness_trainable_weights <- function(mod) all(!duplicated(
  sapply(shape_trainable_weights(mod),
         function(x) paste(x, collapse = "_"))))

get_set_weights <- function(mod_tc, mod_sw){
  for(i in 1:length(mod_tc$model$layers)){
    if(length(mod_tc$model$layers[[i]]$get_weights())>0 && 
       mod_tc$model$layers[[i]]$trainable){
      for(j in 1:length(mod_sw$model$layers)){
        try(
          mod_tc$model$layers[[i]]$set_weights(
            mod_sw$model$trainable_weights[[j]]
          )
          , silent = TRUE)
      }
    }
  }
}

transfer_weights <- function(mod_to_change, mod_supplying_weights){
  
  trainable_tc = shape_trainable_weights(mod_to_change)
  trainable_sw = shape_trainable_weights(mod_supplying_weights)
  
  stc <- sapply(trainable_tc, paste, collapse = "_")
  ssw <- sapply(trainable_sw, paste, collapse = "_")
  
  if(!all(!duplicated(stc)=="TRUE") | !all(!duplicated(ssw)=="TRUE"))
     stop("Ambiguity in the weights, can't transfer.")
  
  if(length(trainable_sw)>length(trainable_tc))
    stop("Can't transfer weigths if trainable weights of supplying model are more.")
  if(length(trainable_sw)==length(trainable_tc)){
    if(length(setdiff(ssw, stc))==0)
    {
      get_set_weights(mod_to_change, mod_supplying_weights)
    }else{
      stop("Same number of trainable layers, but shapes differ.")
    }
  }else{ # length tc larger sw
    if(length(setdiff(ssw, stc))==0){
      get_set_weights(mod_to_change, mod_supplying_weights)
    }else{
      stop("No match between shapes.")
    }
  }
  cat("Done.")
  return(invisible(NULL))
}

reduce_one_list <- function(x)
{
  
  if(is.list(x)) return(x[[1]]) else return(x)
  
}
