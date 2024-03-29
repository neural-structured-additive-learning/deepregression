% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/layers_torch.R
\name{layer_dense_module}
\alias{layer_dense_module}
\title{Function to create custom nn_linear module to overwrite reset_parameters}
\usage{
layer_dense_module(kernel_initializer)
}
\arguments{
\item{kernel_initializer}{string; initializer used to reset_parameters}
}
\value{
nn module
}
\description{
Function to create custom nn_linear module to overwrite reset_parameters
}
