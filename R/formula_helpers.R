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

remove_intercept <- function(form) update(form, ~ 0 + . )

mismatch_brackets <- function(x, logical=TRUE, bracket_set = c("\\{", "\\}"))
{

  open_matches <- lengths(regmatches(x, gregexpr(bracket_set[1], x)))
  close_matches <- lengths(regmatches(x, gregexpr(bracket_set[2], x)))

  if(logical) return(open_matches!=close_matches) else
    return(c(open_matches, close_matches))

}

# used in subnetwork_init
make_valid_layername <- function(string)
{

  gsub("[^a-zA-Z0-9/-]+","_",string)

}

#### helper functions for processors


#' @export
makelayername <- function(term, param_nr, truncate = 60)
{

  if(class(term)=="formula") 
    term <- form2text(term)
  if(grepl("const\\(", term)) 
    term <- gsub("const\\((.*?)\\)", "\\1", term) 
  return(paste0(strtrim(make_valid_layername(gsub("%X%", "", term)), 
                        truncate), "_", param_nr))

}

#' @export
extractvar <- function(term, allow_ia = FALSE)
{

  if(allow_ia){
    pattern <- ".*\\((.*[\\:|\\*].*)\\)"
    org_term <- gsub(pattern, "\\1", term)
    term <- gsub(pattern, "helpervariable123", term)
  }
  
  ret <- all.vars(as.formula(paste0("~", term)))
  
  if(allow_ia){
    ret <- gsub("helpervariable123", org_term, ret)
  }

  return(ret)
}

#' Formula helpers
#'
#' @param term character representing a formula term
#' @param name character; the value to extract
#' @param default_for_missing logical; if TRUE, returns \code{default} if argument is missing
#' @param default value returned when missing
#' @return the value used for \code{name}
#' @export
#' @rdname formulaHelpers
#' @examples
#' extractval("s(a, la = 2)", "la")
#'
extractval <- function(term, name, default_for_missing = FALSE, default = NULL)
{

  if(is.character(term)) term <- as.formula(paste0("~", term))
  inputs <- as.list(as.list(term)[[2]])[-1]
  if(name %in% names(inputs)) return(inputs[[name]])
  if(default_for_missing) return(default)
  warning("Argument ", name, " not found. Setting it to some default.")
  if(name=="df") return(NULL) else if(name=="la") return(0.1) else return(default)

}

# multiple value option of extractval
extractvals <- function(term, names){
  if(is.character(term)) term <- as.formula(paste0("~", term))
  inputs <- as.list(as.list(term)[[2]])[-1]
  return(sapply(names, function(name){
    if(name %in% names(inputs)) inputs[[name]] else NULL
  }))
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

#' @param form formula that is converted to a character string
#' @export
#' @rdname formulaHelpers
form2text <- function(form)
{

  return(gsub(" ","", (Reduce(paste, deparse(form)))))

}

get_special <- function(term, specials, simplify = FALSE)
{

  if(simplify){
    if(term=="(Intercept)") return(NULL)
    if(!grepl("\\(", term)) return(NULL)
    return(gsub("(.*)\\((.*))","\\1",term))
  }
  sp <- attr(terms.formula(as.formula(paste0("~",term)),
                           specials = specials), "specials")
  names(unlist(sp))

}


#' @export
get_processor_name <- function(term)
{

  gsub("([^\\(])\\(.*","\\1", term)

}

get_terms_rwt <- function(term)
{

  trimws(strsplit(gsub("rwt\\((.*)\\)", "\\1", term), split="%X%")[[1]])

}

get_terms_mult <- function(term)
{
 
  term <- gsub("mult\\((.*)\\)", "\\1", term)
  res <- strsplit(term, ",\\s*(?![^()]*\\))", perl=TRUE)[[1]]
  trimws(res)
  
}

remove_layer <- function(term){

  gsub("\\,\\s?layer\\s?=.*[^\\)]","",term)

}

rename_rwt <- function(form){

  tefo <- terms(form)
  trms <- attr(tefo,"term.labels")
  if(length(trms)==0) return(form)
  int <- attr(tefo,"intercept")

  rwts <- grepl("%X%", trms)

  if(all(rwts)) return(form)

  if(any(rwts)){

    trms <- unlist(lapply(trms, function(x){

      if(grepl("%X%", x)){

        if(grepl("^\\(.*\\)\\s?%X%\\s.*?", x))
          x = expand_rwt(x, 1)
        if(grepl(".*\\s?%X%\\s?\\(.*\\)$", x))
          x = expand_rwt(x, 2)

      }
      return(x)

    }))

    rwts <- grepl("%X%", trms) & !grepl("rwt\\(", trms)

    trms[which(rwts)] <- sapply(trms[which(rwts)], function(x){

      return(paste0("rwt(", x, ")"))

    })

  }else{
    
    return(form)
    
  }

  form <- paste(trms, collapse = " + ")
  if(!int) form <- paste0("0 + ", form)
  form <- as.formula(paste0("~ ", form))

  return(form)

}

expand_rwt <- function(x, side){

  if(side==1){
    bracket_terms <- gsub("^\\((.*)\\)\\s?%X%\\s(.*)?", "\\1", x)
    kron_term <- gsub("^\\((.*)\\)\\s?%X%\\s(.*)?", "\\2", x)
  }else{
    kron_term <- gsub("(.*)\\s?%X%\\s?\\((.*)\\)$", "\\1", x)
    bracket_terms <- gsub("(.*)\\s?%X%\\s?\\((.*)\\)$", "\\2", x)
  }

  kron_term <- trimws(kron_term)
  bracket_terms <- trimws(strsplit(bracket_terms, "+", fixed = T)[[1]])
  sapply(bracket_terms, function(b) paste0(kron_term, " %X% ", b))


}


rename_offset <- function(form)
{

  as.formula(gsub("offset\\(", "offsetx\\(", form2text(form)))

}

save_nested_brackets_match <- function(x, start=NULL){
  
  xx <- strsplit(x, "")[[1]]
  if(is.null(start)) start <- which(xx=="(")
  if(length(start)==0) return(x)
  open <- 1
  closing <- 0
  i <- start[1]
  while(open > closing & i < length(xx)){
    i <- i + 1
    if(xx[i]==")") closing <- closing + 1
    if(xx[i]=="(") open <- open + 1
  }
  return(substring(x, start[1], i))
  
}

# extract_kerasoptions <- function(term,
#                                  activation = extractval(term, "activation", TRUE),
#                                  use_bias = extractval(term, "use_bias", TRUE, FALSE),
#                                  trainable = extractval(term, "trainable", TRUE, TRUE),
#                                  kernel_initializer = extractval(term, "kernel_initializer", TRUE, "glorot_uniform"),
#                                  bias_initializer = extractval(term, "bias_initializer", TRUE, "zeros"),
#                                  kernel_regularizer = extractval(term, "kernel_regularizer", TRUE),
#                                  bias_regularizer = extractval(term, "bias_regularizer", TRUE),
#                                  activity_regularizer = extractval(term, "activity_regularizer", TRUE),
#                                  kernel_constraint = extractval(term, "kernel_constraint", TRUE),
#                                  bias_constraint = extractval(term, "bias_constraint", TRUE))
# {
#
#   list(
#     activation = activation,
#     use_bias = use_bias,
#     trainable = trainable,
#     kernel_initializer = kernel_initializer,
#     bias_initializer = bias_initializer,
#     kernel_regularizer = kernel_regularizer,
#     bias_regularizer = bias_regularizer,
#     activity_regularizer = activity_regularizer,
#     kernel_constraint = kernel_constraint,
#     bias_constraint = bias_constraint
#   )
#
# }
