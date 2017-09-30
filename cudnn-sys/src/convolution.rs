#![allow(non_camel_case_types)]

use {c_int, size_t};

use cudnnDataType;
use cudnnStatus;
use cudnnHandle;
use cudnnTensorDescriptor;
use cudnnFilterDescriptor;

pub enum cudnnConvolutionStruct {}
pub type cudnnConvolutionDescriptor = *mut cudnnConvolutionStruct;

#[link(name = "cudnn")]
extern "system" {
    pub fn cudnnCreateConvolutionDescriptor(convDesc: *mut cudnnConvolutionDescriptor)
                                            -> cudnnStatus;
    pub fn cudnnDestroyConvolutionDescriptor(convDesc: cudnnConvolutionDescriptor) -> cudnnStatus;
}

#[repr(C)]
pub enum cudnnConvolutionMode {
    CUDNN_CONVOLUTION = 0,
    CUDNN_CROSS_CORRELATION = 1,
}

#[link(name = "cudnn")]
extern "system" {
    pub fn cudnnSetConvolution2dDescriptor(convDesc: cudnnConvolutionDescriptor,
                                           pad_h: c_int,
                                           pad_w: c_int,
                                           u: c_int,
                                           v: c_int,
                                           dilation_h: c_int,
                                           dilation_w: c_int,
                                           mode: cudnnConvolutionMode,
                                           computeType: cudnnDataType)
                                           -> cudnnStatus;
}

#[link(name = "cudnn")]
extern "system" {
    pub fn cudnnGetConvolution2dForwardOutputDim(convDesc: cudnnConvolutionDescriptor,
                                                 inputTensorDesc: cudnnTensorDescriptor,
                                                 filterDesc: cudnnFilterDescriptor,
                                                 n: *mut c_int,
                                                 c: *mut c_int,
                                                 h: *mut c_int,
                                                 w: *mut c_int)
                                                 -> cudnnStatus;
}

#[repr(C)]
pub enum cudnnConvolutionFwdAlgo {
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT = 4,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = 5,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = 6,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = 7,
    CUDNN_CONVOLUTION_FWD_ALGO_COUNT = 8,
}

#[link(name = "cudnn")]
extern "system" {
    pub fn cudnnGetConvolutionForwardWorkspaceSize(handle: cudnnHandle,
                                                   xDesc: cudnnTensorDescriptor,
                                                   wDesc: cudnnFilterDescriptor,
                                                   convDesc: cudnnConvolutionDescriptor,
                                                   yDesc: cudnnTensorDescriptor,
                                                   algo: cudnnConvolutionFwdAlgo,
                                                   sizeInBytes: *mut size_t)
                                                   -> cudnnStatus;
}
