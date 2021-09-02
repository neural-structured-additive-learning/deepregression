#' Function to loop through parsed formulas and apply data trafo
#' 
#' @param pfc list of processor transformed formulas 
#' @param newdata list in the same format as the original data
#' @return list of matrices or arrays
#' 
loop_through_pfc_and_call_trafo <- function(pfc, newdata = NULL)
{
  
  data_list <- list()
  k <- 1
  for(i in 1:length(pfc))
  {
    
    for(j in 1:length(pfc[[i]])){
      
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
#' @return list of matrices or arrays
#' 
prepare_data <- function(pfc)
{
  
  return(
    loop_through_pfc_and_call_trafo(pfc = pfc, newdata = NULL)
  )
  
}

#' Function to prepare new data based on parsed formulas
#' 
#' @param pfc list of processor transformed formulas 
#' @param newdata list in the same format as the original data
#' @return list of matrices or arrays
#' 
prepare_newdata <- function(pfc, newdata)
{
  
  return(
    loop_through_pfc_and_call_trafo(pfc = pfc, newdata = newdata)
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
