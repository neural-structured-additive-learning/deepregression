# from keras
as_constraint <- getFromNamespace("as_constraint", "keras")
compose_layer <- getFromNamespace("compose_layer", "keras")

make_valid_layername <- function(string)
{
  
  gsub("[^a-zA-Z0-9/-]+","_",string)
  
}

tib_layer = function(units, use_bias = FALSE, la, ...) {
  python_path <- system.file("python", package = "deepregression")
  layers <- reticulate::import_from_path("layers", path = python_path)
  layers$TibLinearLasso(num_outputs = units, use_bias = use_bias, 
                        la = la, ...)
}

# tp and ttp have different positioning internally (b,a instead of a,b)
# as this results in a sparser design and penalty matrix
tp_layer = function(a, b, pen=NULL, name=NULL, units = 1) {
  x <- tf_row_tensor(b, a) %>% 
    layer_dense(units = units, activation = "linear", 
                name = name, use_bias = FALSE, 
                kernel_regularizer = pen)
  return(x)
}

ttp_layer = function(a, b, c, pen=NULL, name=NULL, units = 1) {
  x <- tf_row_tensor(tf_row_tensor(b, c), a) %>% 
    layer_dense(units = units, activation = "linear", 
                name = name, use_bias = FALSE, 
                kernel_regularizer = pen)
  return(x)
}

tf_stride_cols <- function(A, start, end=NULL)
{
  
  if(is.null(end)) end <- start
  return(
    #tf$strided_slice(A, c(0L,as.integer(start-1)), c(tf$shape(A)[1], as.integer(end)))
    tf$keras$layers$Lambda(function(x) x[,as.integer(start):as.integer(end)])(A)
    )
  

}
    
pen_vc <- function(P, strength, nlev)
{
  python_path <- system.file("python", package = "deepregression")
  splines <- reticulate::import_from_path("psplines", path = python_path)
  return(splines$squaredPenaltyVC(P = as.matrix(P), strength=strength, nlev))
  
}

vc_block <- function(ncolNum, levFac, penalty = NULL, name = NULL, units = 1){
  
  if(!is.null(penalty))
    penalty = pen_vc(penalty, strength=1, nlev=levFac)
  
  ret_fun <- function(x){
    
    a = tf_stride_cols(x, 1, ncolNum)
    b = tf$one_hot(tf$cast(
      (tf_stride_cols(x, ncolNum+1)[,1]), 
      dtype="int32"), 
      depth=levFac)
    return(tp_layer(a, b, pen=penalty, name=name, units=units))
    
  }
  return(ret_fun)
}

vvc_block <- function(ncolNum, levFac1, levFac2, penalty = NULL, name = NULL, units = 1){
  
  if(!is.null(penalty))
    penalty = pen_vc(penalty, strength=1, nlev=levFac1*levFac2)
  
  ret_fun <- function(x) ttp_layer(x[,as.integer(1:ncolNum)],
                                  tf$one_hot(tf$cast(x[,as.integer((ncolNum+1))], dtype="int32"), 
                                             depth=levFac1),
                                  tf$one_hot(tf$cast(x[,as.integer((ncolNum+2))], dtype="int32"), 
                                             depth=levFac2), pen=penalty, name=name, units=units)
  
  return(ret_fun)
}

pen_vc_gen <- function()

layer_factor <- function(nlev, units = 1, activation = "linear", use_bias = FALSE, name = NULL,
                         kernel_regularizer = NULL)
{
  
  ret_fun <- function(x) tf$one_hot(tf$cast(x[,1], dtype="int32"), depth = nlev) %>% layer_dense(
    units = units,
    activation = activation,
    use_bias = use_bias,
    name = name,
    kernel_regularizer = kernel_regularizer)
  return(ret_fun)
  
}

layer_random_effect <- function(freq, df)
{
  
  df_fun <- function(lam) sum((freq^2 + 2*freq*lam)/(freq+lam)^2)
  lambda = uniroot(function(x){df_fun(x)-df}, interval = c(0,1e15))$root
  nlev = length(freq)
  return(
    layer_factor(nlev = nlev, units = 1, activation = "linear", use_bias = FALSE, name = NULL,
                 kernel_regularizer = regularizer_l2(l = lambda/sum(freq)))
  )
  
}
