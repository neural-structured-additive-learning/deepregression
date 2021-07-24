#' Function to prepare data based on parsed formulas
#' 
#' @param pfc list of processor transformed formulas 
#' @return list of matrices or arrays
#' 
prepare_data <- function(pfc)
{
  
  data_list <- list()
  k <- 1
  for(i in 1:length(pfc))
  {
    
    for(j in 1:length(pfc[[i]])){
      
      data_list[[k]] <- to_matrix(pfc[[i]][[j]]$data_trafo())
      k <- k + 1
      
    }
    
  }
  
  return(data_list)
  
}

#' Function to prepare new data based on parsed formulas
#' 
#' @param pfc list of processor transformed formulas 
#' @param newdata list in the same format as the original data
#' @return list of matrices or arrays
#' 
prepare_newdata <- function(pfc, newdata)
{
  
  data_list <- list()
  k <- 1
  for(i in 1:length(pfc))
  {
    
    for(j in 1:length(pfc[[i]])){
      
      data_list[[k]] <- to_matrix(pfc[[i]][[j]]$predict_trafo(newdata))
      k <- k + 1
      
    }
    
  }
  
  return(data_list)
  
}

to_matrix <- function(x)
{
  
  if(is.list(x)) return(do.call("cbind", x))
  if(is.data.frame(x)) return(as.matrix(x))
  return(x)
  
}
