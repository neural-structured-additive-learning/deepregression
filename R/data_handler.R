#' Function to loop through parsed formulas and apply data trafo
#' 
#' @param pfc list of processor transformed formulas 
#' @param newdata list in the same format as the original data
#' @param engine character; the engine which is used to setup the NN (tf or torch)
#' @return list of matrices or arrays
#' 
loop_through_pfc_and_call_trafo <- function(pfc, newdata = NULL, engine = "tf")
{
  
  data_list <- list()
  k <- 1
  for(i in 1:length(pfc))
  {
    
    for(j in 1:length(pfc[[i]])){
      
      # skip those which are already set up by the gamdata
      if(!is.null(pfc[[i]][[j]]$gamdata_nr) & engine == 'tf')
        if(!pfc[[i]][[j]]$gamdata_combined) next
      
      if(is.null(newdata)){
        data_list[[k]] <- to_matrix(pfc[[i]][[j]]$data_trafo())
      }else{
        data_list[[k]] <- to_matrix(pfc[[i]][[j]]$predict_trafo(newdata))
  }
      k <- k + 1
    }
      
      
  }
  
  return(data_list)
  
  
}

#' Function to prepare data based on parsed formulas
#' 
#' @param pfc list of processor transformed formulas 
#' @param na_handler function to deal with NAs
#' @param gamdata processor for gam part
#' @param engine the engine which is used to setup the NN (tf or torch)
#' @return list of matrices or arrays
#' @export
#' 
prepare_data <- function(pfc, na_handler = na_omit_list, gamdata = NULL,
                         engine = "tf")
{
  
  ret_list <- loop_through_pfc_and_call_trafo(pfc = pfc, engine = engine)
  if(!is.null(gamdata) & engine == "tf")
    ret_list <- c(prepare_gamdata(gamdata), ret_list)
  
  ret_list <- na_handler(ret_list)
  
  return(ret_list)
  
}

#' Function to prepare new data based on parsed formulas
#' 
#' @param pfc list of processor transformed formulas 
#' @param na_handler function to deal with NAs
#' @param newdata list in the same format as the original data
#' @param gamdata processor for gam part
#' @param engine character; the engine which is used to setup the NN (tf or torch)
#' @return list of matrices or arrays
#' @export
#' 
prepare_newdata <- function(pfc, newdata, na_handler = na_omit_list, gamdata = NULL,
                            engine = "tf")
{
  
  ret_list <- loop_through_pfc_and_call_trafo(pfc = pfc, newdata = newdata,
                                              engine = engine)
  
  if(!is.null(gamdata) & engine == 'tf')
    ret_list <- c(prepare_gamdata(gamdata, newdata), ret_list)
  ret_list <- na_handler(ret_list)
  
  return(ret_list)
  
}


prepare_gamdata <- function(gamdata, newdata = NULL){
  
  if(is.null(newdata))
    return(
      unname(lapply(gamdata, function(x) 
        to_matrix(x$data_trafo())))
    )
  
  return(
    unname(lapply(gamdata, function(x) 
      to_matrix(x$predict_trafo(newdata))))
  )
}

to_matrix <- function(x)
{
  
  if(is.list(x)){ 
    if(length(x)==1 & !is.null(dim(x[[1]]))){ # array as input
      return(x[[1]])
    }else{
      return(do.call("cbind", x))
    }
  }
  if(is.data.frame(x)) return(as.matrix(x))
  return(x)
  
}

#' Function to exclude NA values
#' 
#' @param datalist list of data as returned by \code{prepare_data} and 
#' \code{prepare_newdata}
#' @return list with NA values excluded and locations of original
#' NA positions as attributes
#' @export
#' 
na_omit_list <- function(datalist)
{
  
  na_loc <- unique(unlist(lapply(datalist, function(x) 
    unique(apply(x, 2, function(y) which(is.na(y)))))))
  
  if(length(na_loc) > 0)
    datalist <- lapply(datalist, function(x) x[-na_loc,,drop=FALSE])
  attr(datalist, "na_loc") <- na_loc
  
  return(datalist)
  
}