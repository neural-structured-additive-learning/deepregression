# needed by gam_processor
#' @export
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

# needed by gam_processor
#' @export
orthog_P <- function(P,Z)
{
  return(crossprod(Z,P) %*% Z)
}

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

#' Function to define orthogonalization connections in the formula
#' 
#' @param form a formula for one distribution parameter
#' @param specials specials in formula to handle separately
#' @param specials_to_oz parts of the formula to orthogonalize
#' @param automatic_oz_check logical; automatically check if terms must be orthogonalized
#' @param identify_intercept logical; whether to make the intercept identifiable
#' @param simplify logical; if FALSE, formulas are parsed more carefully.
#' @return Returns a list of formula components with ids and 
#' assignments for orthogonalization
#' 
#' 
separate_define_relation <- function(
  form, 
  specials, 
  specials_to_oz, 
  automatic_oz_check = TRUE,
  identify_intercept = FALSE,
  simplify = FALSE
  )
{
  
  if(simplify){
    terms <- trimws(strsplit(as.character(form)[[2]], split = "+", fixed = TRUE)[[1]])
    terms <- terms[terms!=""]
    terms <- lapply(1:length(terms), function(i) list(term = terms[i], nr = i,
                                                      left_from_oz = TRUE,
                                                      right_from_oz = NULL))
    terms <- append(terms,
                    list(list(
                      term = "(Intercept)",
                      nr = length(terms)+1,
                      left_from_oz = TRUE,
                      right_from_oz = NULL
                    )))
    return(terms)
    
  }
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
        if(has_intercept & identify_intercept) these_terms <- c("1", these_terms)
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
        is_already_right <- is_equal_not_null(terms_right[[i]][[k]], 
                                              sapply(add_terms, "[[", "term"))
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
