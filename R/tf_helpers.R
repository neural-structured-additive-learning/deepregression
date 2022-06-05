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
  
  stopifnot(start <= end)
  if(is.null(end)) end <- start
  return(
    #tf$strided_slice(A, c(0L,as.integer(start-1)), c(tf$shape(A)[1], as.integer(end)))
    tf$keras$layers$Lambda(function(x) x[,as.integer(start):as.integer(end)])(A)
  )
  
  
}

#' Function to index tensors last dimension
#' 
#' @param A tensor
#' @param start first index
#' @param end last index (equals start index if NULL)
#' @return sliced tensor
#' @export
tf_stride_last_dim_tensor <- function(A, start, end=NULL){

  stopifnot(start <= end)
  if(is.null(end)) end <- start
  mat <- as.integer(A$shape)
  sz <- mat
  sz[length(sz)] <- end-start+1L
  return(
    tf$slice(A, begin = as.integer(c(rep(0, length(mat)-1), start-1L)),
             size = as.integer(sz))
             
  )
  
}

#' @export
#'
#'
tf_split_multiple <- function(A, len){
  
  ends <- cumsum(len)
  starts <- c(1, ends[-length(ends)]+1)
  lapply(1:length(starts), function(i) tf_stride_cols(A, starts[i], ends[i]))
  
}

# function to convert constant to TF float32 tensor
convertfun_tf <- function(x) tf$constant(x, dtype="float32")

#' TensorFlow repeat function which is not available for TF 2.0
#' 
#' @param a tensor
#' @param dim dimension for repeating
#' 
#' @export
#' 
tf_repeat <- function(a, dim)
  tf$reshape(tf$tile(tf$expand_dims(a, axis = -1L),  c(1L, 1L, dim)), 
             shape = list(-1L, a$shape[[2]]*dim))

#' Row-wise tensor product using TensorFlow
#' 
#' @param a,b tensor
#' 
#' @export
#' 
tf_row_tensor <- function(a, b, ...)
{
  # tf$multiply(
  #   tf_row_tensor_left_part(a,b),
  #   tf_row_tensor_right_part(a,b)
  # )
  python_path <- system.file("python", package = "deepregression")
  misc <- reticulate::import_from_path("misc", path = python_path)
  misc$RowTensor(...)(list(a, b))
}

tf_row_tensor_left_part <- function(a,b)
{
  tf_repeat(a, b$shape[[2]])
}

tf_row_tensor_right_part <- function(a,b)
{
  tf$tile(b, c(1L, a$shape[[2]]))
}
