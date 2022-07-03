#' used by gam_processor
#' @param pp processed term
#' @param weights layer weights
#' @param grid_length length for grid for evaluating basis
#' @param pe_fun function used to generate partial effects
#' @export
gam_plot_data <- function(pp, weights, grid_length = 40, pe_fun = pe_gen)
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
    if(is.factor(org_values[[2]])){
      this_y <- unique(org_values[[2]])
    }else{
      this_y <- do.call(seq, c(as.list(range(plotData$value[,2])),
                               list(l=grid_length)))
    }
    df <- as.data.frame(expand.grid(this_x, this_y))
    colnames(df) <- extractvar(pp$term)
    plotData$df <- df
    plotData$x <- this_x
    plotData$y <- this_y
    plotData$partial_effect <- pe_fun(pp, df, weights)

  }else{

    warning("Plot for more than 2 dimensions not implemented yet.")

  }

  return(plotData)

}

pe_gen <- function(pp, df, weights){

  pmat <- pp$predict_trafo(newdata = df)
  pmat%*%weights

}

layer_spline = function(units = 1L, P, name, trainable = TRUE,
                        kernel_initializer = "glorot_uniform") {
  python_path <- system.file("python", package = "deepregression")
  splines <- reticulate::import_from_path("psplines", path = python_path)

  return(splines$layer_spline(P = as.matrix(P), units = units, trainable = trainable,
                              name = name, kernel_initializer = kernel_initializer))
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

#' Convenience function to extract penalty matrix and value
#' @param x evaluated smooth term object
#' @export
extract_S <- function(x)
{

  if((length(x[[1]]$S)==1 & is.null(x[[1]]$by.level)) |
     (length(x[[1]]$S)==2 & !is.null(length(x[[1]]$margin)))){

    S <- x[[1]]$S

  }else{

    S <- lapply(x,function(y) y$S[[1]])

  }

  return(S = S)

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
      sp <- sapply(1:length(st[[1]]$margin), function(i)
      {
        DRO(st[[1]]$margin[[i]]$X,
            df = this_df[i],
            dmat = st[[1]]$margin[[i]]$S[[1]],
            hat1 = hat1
        )["lambda"] +
          null_space_penalty
      })
      return(sp)
    }else{
      X <- do.call("cbind", lapply(st,"[[","X"))
    }
  }
  sp = DRO(X, df = this_df, dmat = S, hat1 = hat1)["lambda"] +
    null_space_penalty
  return(sp)
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
  object <- remove_df_wrapper(object)
  names_s <- all.vars(as.formula(paste0("~", object)))
  sterm <- smoothCon(eval(parse(text=object)),
                     data=data.frame(data[names_s]),
                     absorb.cons = controls$absorb_cons,
                     null.space.penalty = controls$null_space_penalty
  )
  # sterm <- controls$defaultSmoothing(sterm, df)
  return(sterm)

}

remove_df <- function(object)
{

  gsub(",\\s?df\\s?=\\s?[0-9.-]+","",object)

}

remove_zerocons <- function(object)
{

  gsub(",\\s?zerocons\\s?=\\s?(TRUE|FALSE)","",object)

}

remove_la <- function(object)
{

  gsub(",\\s?la\\s?=\\s?(-?[0-9]*)((\\.?[0-9]*[eE]?[-\\+]?[0-9]+)|(\\.[0-9]+))",
       "",object)

}


predict_gam_handler <- function(object, newdata)
{

  if(is.list(object) && length(object)==1) return(PredictMat(object[[1]], as.data.frame(newdata[object[[1]]$term])))
  return(do.call("cbind", lapply(object, function(obj) PredictMat(obj, as.data.frame(newdata))))  )

}

#' Extract gam part from wrapped term
#'
#' @param term character; gam model term
#' @param wrapper character; function name that is wrapped around the gam part
#'
#' @export
get_gam_part <- function(term, wrapper = "vc")
{

  gsub(paste0(wrapper, "\\(((s|te|ti)\\(.*\\))\\,\\s*by\\s*=.*\\)"),"\\1", term)

}

squaredPenalty <- function(P, strength)
{
  python_path <- system.file("python", package = "deepregression")
  splines <- reticulate::import_from_path("psplines", path = python_path)

  return(splines$squaredPenalty(P = as.matrix(P), strength = strength))

}

remove_df_wrapper <- function(object){

  df <- suppressWarnings(extractval(object, "df"))
  if(!is.null(df)){
    return(remove_df(object))
  }else{
    return(object)
  }

}

remove_zerocons_wrapper <- function(object){

  zerocons <- suppressWarnings(extractval(object, "zerocons"))
  if(!is.null(zerocons)){
    return(remove_zerocons(object))
  }else{
    return(object)
  }

}

remove_la_wrapper <- function(object){

  la <- suppressWarnings(extractval(object, "la"))
  if(!is.null(la)){
    return(remove_la(object))
  }else{
    return(object)
  }

}

fix_bracket_mismatch <- function(x)
{

  open_matches <- lengths(regmatches(x, gregexpr("\\(", x)))
  close_matches <- lengths(regmatches(x, gregexpr("\\)", x)))
  if(open_matches==close_matches) return(x)
  if(open_matches < close_matches) stop("Too many closing brackets")
  if(open_matches > close_matches)
    fix_bracket_mismatch(paste0(x,")"))

}

#'
#' @export
extract_pure_gam_part <- function(term, remove_other_options=TRUE){

  term <- gsub(".*(^|\\()(s|te|ti)\\((.*)", "\\2(\\3", term)
  term <- gsub("(s|te|ti)(\\()(.*?)(\\))(.*)", "\\1\\2\\3\\4", term)
  term <- fix_bracket_mismatch(term)
  if(remove_other_options){
    term <- remove_df_wrapper(term)
    term <- remove_zerocons_wrapper(term)
    term <- remove_la_wrapper(term)
  }

  return(term)

}

create_data_trafos <- function(evaluated_gam_term, controls, xlin)
{

  # extract Xs
  if(length(evaluated_gam_term)==1){
    thisX <- evaluated_gam_term[[1]]$X
  }else{
    thisX <- do.call("cbind", lapply(evaluated_gam_term, "[[", "X"))
  }
  # get default Z matrix, which is possibly overwritten afterwards
  Z <- diag(rep(1,ncol(thisX)))
  # constraint
  if((controls$zero_constraint_for_smooths | 
      controls$no_linear_trend_for_smooths ) &
     length(evaluated_gam_term)==1 &
     !evaluated_gam_term[[1]]$dim>1){
    
    conMat <- list()
    
    if(controls$zero_constraint_for_smooths)
      conMat[[1]] <- matrix(rep(1,NROW(evaluated_gam_term[[1]]$X)), ncol=1)
    if(controls$no_linear_trend_for_smooths)
      conMat[[2]] <- xlin
    conMat <- do.call("cbind", conMat)
    
    Z <- orthog_structured_smooths_Z(
      evaluated_gam_term[[1]]$X,
      conMat
    )
    
  }

  return(
    list(data_trafo = function() thisX %*% Z,
         predict_trafo = function(newdata) predict_gam_handler(evaluated_gam_term,
                                                               newdata = newdata) %*% Z,
         input_dim = as.integer(ncol(thisX %*% Z)),
         partial_effect = function(weights, newdata=NULL){
           if(is.null(newdata))
             return(thisX %*% Z %*% weights)
           return(predict_gam_handler(evaluated_gam_term, newdata = newdata) %*% Z %*% weights)
         },
         Z = Z)
  )

}

create_penalty <- function(evaluated_gam_term, df, controls, Z = NULL)
{

  # get sp and S
  sp_and_S <- list(
    sp = controls$defaultSmoothing(evaluated_gam_term, df),
    S = extract_S(evaluated_gam_term)
  )

  if(controls$zero_constraint_for_smooths &
     length(evaluated_gam_term)==1 &
     !evaluated_gam_term[[1]]$dim>1 & !is.null(Z)){

    sp_and_S[[2]][[1]] <- orthog_P(sp_and_S[[2]][[1]],Z)

  }else if(evaluated_gam_term[[1]]$dim>1 &
           length(evaluated_gam_term)==1){
    # tensor product -> merge and keep dummy
    sp_and_S <- list(sp = 1,
                     S = list(do.call("+", lapply(1:length(sp_and_S[[2]]), function(i)
                       sp_and_S[[1]][i] * sp_and_S[[2]][[i]]))))
  }

  return(list(sp_and_S = sp_and_S))

}

#' Pre-calculate all gam parts from the list of formulas
#' @param lof list of formulas
#' @param data the data list
#' @param controls controls from deepregression
#'
#' @return a list of length 2 with a matching table to
#' link every unique gam term to formula entries and the respective
#' data transformation functions
#' @export
#'
precalc_gam <- function(lof, data, controls)
{

  tfs <- lapply(lof, function(form) terms.formula(form, specials = c("s", "te", "ti")))
  termstrings <- lapply(tfs, function(tf) trmstrings <- attr(tf, "term.labels"))
  # split % % parts
  termstrings <- lapply(termstrings, function(tfs) unlist(sapply(tfs, function(tf)
    trimws(strsplit(tf, split = "%.*%")[[1]]))))
  # split terms in brackets and remove leading brackets
  termstrings <- lapply(termstrings, function(tfs) unlist(sapply(tfs, function(tf)
    gsub("^\\(", "", trimws(strsplit(tf, split = "+", fixed = T)[[1]])))))
  gam_terms <- lapply(termstrings, function(tf) tf[grepl("(^|\\()(s|te|ti)\\(", tf) | 
                                                     grepl("~\\s?(s|te|ti)\\(", tf)])
  gam_terms <- lapply(gam_terms, function(tf){ 
    if(any(grepl("~", tf))){
      return(
        c(tf[!grepl("~", tf)], sapply(tf[grepl("~", tf)], function(x){
          
          parts <- strsplit(x, "~")[[1]][-1]
          bracket_mismatch <- mismatch_brackets(parts, bracket_set = c("\\(", "\\)"))
          parts[bracket_mismatch] <- gsub("\\)\\)$", ")", parts[bracket_mismatch])
          return(parts)
          
        })) 
        )
      }else return(tf)
    })
  
  # bracket_mismatch <- mismatch_brackets(ul_gam_terms,
  #                                       bracket_set = c("\\(", "\\)"))
  # ul_gam_terms[bracket_mismatch] <- gsub("\\)\\)$", ")", ul_gam_terms[bracket_mismatch])
  gam_terms <- lapply(gam_terms, function(gts)
    unlist(sapply(gts, function(gt) sapply(gt, function(gtt) 
      gsub(" ", "", extract_pure_gam_part(gtt, FALSE),
                                  fixed = TRUE)
      )))
    )
  gam_terms <- lapply(gam_terms, unique)
  if(all(sapply(gam_terms, length)==0)) return(NULL)
  len_lists <- rep(1:length(lof), sapply(gam_terms, length))
  ul_gam_terms <- unlist(gam_terms)

  if(length(controls$zero_constraint_for_smooths) > 1)
    stop("Different Zero-Constraints for different Predictors not implemented yet.")

  if(!is.null(controls$df) && length(controls$df)==1)
    controls$df <- rep(controls$df, length(gam_terms))

  # create relevant matching table
  matching_table <- lapply(1:length(ul_gam_terms), function(i){

    df <- suppressWarnings(extractval(ul_gam_terms[i], "df"))
    if(is.null(df)) df <- controls$df[len_lists[i]]
    zerocons <- suppressWarnings(extractval(ul_gam_terms[i], "zerocons"))
    if(is.null(zerocons)) zerocons <- controls$zero_constraint_for_smooths

    return(
      list(
        term = ul_gam_terms[i],
        param_nr = len_lists[i],
        reduced_gam_term = extract_pure_gam_part(ul_gam_terms[i]),
        zero_constraint = zerocons,
        df = df
      )
    )
  })
  names(matching_table) <- ul_gam_terms

  unique_gam_terms <- unique(
    paste0(
      sapply(matching_table, "[[", "reduced_gam_term"),
      "+",
      sapply(matching_table, "[[", "zero_constraint"))
  )

  unique_gam_terms_wo_zc <- unique(
    sapply(matching_table, "[[", "reduced_gam_term")
  )

  sterms <- lapply(unique_gam_terms_wo_zc, function(term)
    handle_gam_term(object = term,
                    data = data,
                    controls = controls)
  )
  names(sterms) <- unique_gam_terms_wo_zc

  data_trafos <- lapply(unique_gam_terms, function(ugt){
    parts <- strsplit(ugt,"+",fixed=T)[[1]]
    w <- which(names(sterms)==parts[1])
    sterm <- sterms[[w]]
    controls$zero_constraint_for_smooths <- as.logical(parts[2])
    xlin <- NULL
    if(length(sterm)==1 & length(sterm[[1]]$term)==1) xlin <- data[[sterm[[1]]$term]]
    ret <- create_data_trafos(sterm, controls, xlin)
    ret$term <- ugt
    return(ret)
  })
  names(data_trafos) <- unique_gam_terms

  for(i in 1:length(matching_table)){
    controls$zero_constraint_for_smooths <- as.logical(matching_table[[i]]$zero_constraint)

    w <- which(names(sterms) == matching_table[[i]]$reduced_gam_term)

    this_penalty <- create_penalty(sterms[[w]],
                                   df = matching_table[[i]]$df,
                                   controls = controls,
                                   Z = data_trafos[[w]]$Z)
    matching_table[[i]] <- c(matching_table[[i]], this_penalty)
  }

  rm(sterms)

  return(list(matching_table = matching_table,
              data_trafos = data_trafos))

}

get_gamdata_nr <- function(term, param_nr, gamdata)
{

  which(sapply(gamdata$matching_table, "[[", "term") == gsub(" ", "", term, fixed=T) &
          sapply(gamdata$matching_table, "[[", "param_nr") == param_nr
  )

}

#' @export
get_gamdata_reduced_nr <- function(term, param_nr, gamdata){

  gamdata_nr <- get_gamdata_nr(term, param_nr, gamdata)

  which(paste0(gamdata$matching_table[[gamdata_nr]]$reduced_gam_term, "+",
               gamdata$matching_table[[gamdata_nr]]$zero_constraint) ==
          names(gamdata$data_trafos))

}

#' @export
get_gamdata <- function(term, param_nr, gamdata,
                        what = c("data_trafo",
                                 "predict_trafo",
                                 "input_dim",
                                 "partial_effect",
                                 "sp_and_S",
                                 "df")){

  if(what %in% c("sp_and_S", "df"))
    return(gamdata$matching_table[[
      get_gamdata_nr(term, param_nr, gamdata)
    ]][[what]])
  # else
  return(
    gamdata$data_trafos[[
      get_gamdata_reduced_nr(term, param_nr, gamdata)
      ]][[what]]
  )

}

create_P <- function(sp_and_S, scale)
{

  as.matrix(bdiag(lapply(1:length(sp_and_S[[1]]), function(i)
    scale * sp_and_S[[1]][[i]] * sp_and_S[[2]][[i]])))

}
