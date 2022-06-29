# from mlr3keras

#' Make a DataGenerator from a data.frame or matrix
#'
#' Creates a Python Class that internally iterates over the data.
#' @param x matrix;
#' @param y vector;
#' @param generator generator as e.g. obtained from `keras::image_data_generator`.
#'   Used for consistent train-test splits.
#' @param batch_size integer 
#' @param shuffle logical; Should data be shuffled?
#' @param seed integer; seed for shuffling data.
#' @export
make_generator_from_matrix = function(x, y = NULL, generator=image_data_generator(), 
                                      batch_size=32L, shuffle=TRUE, seed=1L) {
  python_path <- system.file("python", package = "deepregression")
  generators <- reticulate::import_from_path("generators", path = python_path)
  generators$Numpy2DArrayIterator(x, y, generator, batch_size=as.integer(batch_size), 
                                  shuffle=shuffle,seed=as.integer(seed))
}


combine_generators = function(genList) {
  python_path <- system.file("python", package = "deepregression")
  generators <- reticulate::import_from_path("generators", path = python_path)
  generators$CombinedGenerator(genList)
}

combine_generators_wo_y = function(genList) {
  python_path <- system.file("python", package = "deepregression")
  generators <- reticulate::import_from_path("generators", path = python_path)
  generators$CombinedGeneratorWoY(genList)
}

######################

#' creates a generator for training
#'
#' @param input_x list of matrices
#' @param input_y list of matrix
#' @param batch_size integer
#' @param sizes sizes of the image including colour channel
#' @param shuffle logical for shuffling data
#' @param seed seed for shuffling in generators
#' @return generator for all x and y
make_generator <- function(
  input_x,
  input_y = NULL,
  batch_size,
  sizes,
  shuffle = TRUE,
  seed = 42L
)
{
  
  generators_x <- list()
  j <- 1
  
  for(i in 1:length(input_x)){
    
    if(is.character(input_x[[i]])){

      input_x[[i]] <- as.data.frame(input_x[[i]])
      
      generators_x[[i]] <- flow_images_from_dataframe(input_x[[i]], 
                                                      x_col = colnames(input_x[[i]]), 
                                                      class_mode = NULL,
                                                      target_size = sizes[[j]][1:2],
                                                      color_mode = ifelse(sizes[[j]][3]==3, 
                                                                          "rgb", "grayscale"),
                                                      batch_size = batch_size, 
                                                      shuffle = shuffle, 
                                                      seed = seed)
      
      j <- j + 1
      
    }else{
      
      generators_x[[i]] <- make_generator_from_matrix(
        x = input_x[[i]], 
        y = NULL, 
        batch_size = batch_size, 
        shuffle = shuffle, 
        seed = seed
      ) 
      
    }
    
  }
  
  if(!is.null(input_y)){
    
    generators_y <- make_generator_from_matrix(
      x = input_y,
      y = NULL,
      batch_size = batch_size, 
      shuffle = shuffle, 
      seed = seed
    )
  
    combined_gen <- combine_generators(c(generators_x, list(generators_y)))
    
    # str(combined_gen$`__getitem__`(1L),1)
    
  }else{
    
    combined_gen <- combine_generators_wo_y(generators_x)
    
  }
  
  return(combined_gen)
  
}

prepare_generator_deepregression <- function(
  x, 
  input_x,
  input_y,
  sizes,
  batch_size = 32,
  epochs = 10,
  verbose = TRUE,
  view_metrics = FALSE,
  validation_data = NULL,
  validation_split = 0.1,
  callbacks = list(),
  ...
)
{
  

  if(validation_split==0 | is.null(validation_split) | !is.null(validation_data))
  {
    
    # only fit generator
    max_data <- NROW(input_x[[1]])
    steps_per_epoch <- ceiling(max_data/batch_size)
    
    generator <- make_generator(input_x,
                                input_y,
                                batch_size, 
                                sizes = sizes)
    
    if(!is.null(validation_data)){

      max_data <- NROW(validation_data[[1]][[1]])

      validation_data <- make_generator(validation_data[[1]],
                                        validation_data[[2]],
                                        batch_size, 
                                        sizes = sizes)
      
      validation_steps <- ceiling(max_data/batch_size)
      
    }else{
      
      validation_data <- NULL
      validation_steps <- NULL
      
    }
    
    
  }else{
    
    input_x <- lapply(input_x, as.array)
    
    ind_val <- sample(1:NROW(input_y), round(NROW(input_y)*validation_split))
    ind_train <- setdiff(1:NROW(input_y), ind_val)
    input_x_train <- subset_input_cov(input_x, ind_train)
    input_x_val <- subset_input_cov(input_x, ind_val)
    input_y_train <- matrix(subset_array(input_y, ind_train), ncol=1)
    input_y_val <- matrix(subset_array(input_y, ind_val), ncol=1)
    
    max_data_train <- NROW(input_x_train[[1]])
    steps_per_epoch <- ceiling(max_data_train/batch_size)
    
    generator <- make_generator(input_x_train,
                                input_y_train,
                                batch_size = batch_size, 
                                sizes = sizes)
    
    max_data_val <- NROW(input_x_val[[1]])
    validation_steps <- ceiling(max_data_val/batch_size)

    validation_data <- make_generator(input_x_val,
                                      input_y_val,
                                      batch_size = batch_size, 
                                      sizes = sizes
                                      )
    
  }
  
  args <- list(...)
  args <- c(args, list(
    object = x,
    x = generator,
    epochs = epochs,
    steps_per_epoch = as.integer(steps_per_epoch),
    validation_data = validation_data,
    validation_steps = as.integer(validation_steps),
    callbacks = callbacks,
    verbose = verbose,
    view_metrics = view_metrics
  ))
  
  return(args)
  
}

predict_gen <- function(
  object,
  newdata = NULL,
  batch_size = NULL,
  apply_fun = tfd_mean,
  convert_fun = as.matrix,
  ret_dist = FALSE
)
{
  
  if(!is.null(newdata)){
    newdata_processed <- prepare_newdata(object$init_params$parsed_formulas_contents, 
                                         newdata, 
                                         gamdata = object$init_params$gamdata$data_trafos)
  }else{
    newdata_processed <- prepare_data(object$init_params$parsed_formulas_contents,
                                      gamdata = object$init_params$gamdata$data_trafos)
  }
  # prepare generator
  max_data <- NROW(newdata_processed[[1]])
  if(is.null(batch_size)) batch_size <- 20
  steps_per_epoch <- ceiling(max_data/batch_size)
  
  generator <- make_generator(input_x = newdata_processed,
                              input_y = NULL,
                              batch_size = batch_size,
                              sizes = object$init_params$image_var,
                              shuffle = FALSE)
  
  if(is.null(apply_fun)){ 
    
    apply_fun <- function(x){x}
    ret_dist <- TRUE
    
  }else{
    
    ret_dist <- FALSE
    
  }
  
  res <- lapply(1:steps_per_epoch, function(i) 
    convert_fun(apply_fun(suppressWarnings(
      object$model(generator$`__getitem__`(as.integer(i-1)))))))
  
  if(ret_dist) return(res) else return(do.call("rbind", (res)))
  
}
