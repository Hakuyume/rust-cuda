#![allow(non_camel_case_types)]

use {c_float, c_int, c_void, size_t};

use cudnnDataType;
use cudnnDeterminism;
use cudnnStatus;
use cudnnHandle;
use cudnnTensorDescriptor;
use cudnnFilterDescriptor;

pub enum cudnnConvolutionStruct {}
pub type cudnnConvolutionDescriptor = *mut cudnnConvolutionStruct;

#[repr(C)]
pub enum cudnnConvolutionMode {
    CUDNN_CONVOLUTION = 0,
    CUDNN_CROSS_CORRELATION = 1,
}

#[repr(C)]
pub enum cudnnConvolutionFwdPreference {
    CUDNN_CONVOLUTION_FWD_NO_WORKSPACE = 0,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST = 1,
    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2,
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

#[repr(C)]
pub struct cudnnConvolutionFwdAlgoPerf {
    pub algo: cudnnConvolutionFwdAlgo,
    pub status: cudnnStatus,
    pub time: c_float,
    pub memory: size_t,
    pub determinism: cudnnDeterminism,
    reserved: [c_int; 4],
}

#[repr(C)]
pub enum cudnnConvolutionBwdFilterPreference {
    CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE = 0,
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST = 1,
    CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2,
}

#[repr(C)]
pub enum cudnnConvolutionBwdFilterAlgo {
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = 1,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = 2,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = 3,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD = 4,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = 5,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING = 6,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT = 7,
}

#[link(name = "cudnn")]
extern "system" {
    pub fn cudnnCreateConvolutionDescriptor(convDesc: *mut cudnnConvolutionDescriptor)
                                            -> cudnnStatus;
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
    pub fn cudnnGetConvolution2dForwardOutputDim(convDesc: cudnnConvolutionDescriptor,
                                                 inputTensorDesc: cudnnTensorDescriptor,
                                                 filterDesc: cudnnFilterDescriptor,
                                                 n: *mut c_int,
                                                 c: *mut c_int,
                                                 h: *mut c_int,
                                                 w: *mut c_int)
                                                 -> cudnnStatus;
    pub fn cudnnDestroyConvolutionDescriptor(convDesc: cudnnConvolutionDescriptor) -> cudnnStatus;
    pub fn cudnnFindConvolutionForwardAlgorithm(handle: cudnnHandle,
                                                xDesc: cudnnTensorDescriptor,
                                                wDesc: cudnnFilterDescriptor,
                                                convDesc: cudnnConvolutionDescriptor,
                                                yDesc: cudnnTensorDescriptor,
                                                requestedAlgoCount: c_int,
                                                returnedAlgoCount: *mut c_int,
                                                perfResults: *mut cudnnConvolutionFwdAlgoPerf)
                                                -> cudnnStatus;
    pub fn cudnnGetConvolutionForwardAlgorithm(handle: cudnnHandle,
                                               xDesc: cudnnTensorDescriptor,
                                               wDesc: cudnnFilterDescriptor,
                                               convDesc: cudnnConvolutionDescriptor,
                                               yDesc: cudnnTensorDescriptor,
                                               preference: cudnnConvolutionFwdPreference,
                                               memoryLimitInbytes: size_t,
                                               algo: *mut cudnnConvolutionFwdAlgo)
                                               -> cudnnStatus;
    pub fn cudnnGetConvolutionForwardWorkspaceSize(handle: cudnnHandle,
                                                   xDesc: cudnnTensorDescriptor,
                                                   wDesc: cudnnFilterDescriptor,
                                                   convDesc: cudnnConvolutionDescriptor,
                                                   yDesc: cudnnTensorDescriptor,
                                                   algo: cudnnConvolutionFwdAlgo,
                                                   sizeInBytes: *mut size_t)
                                                   -> cudnnStatus;
    pub fn cudnnConvolutionForward(handle: cudnnHandle,
                                   alpha: *const c_void,
                                   xDesc: cudnnTensorDescriptor,
                                   x: *const c_void,
                                   wDesc: cudnnFilterDescriptor,
                                   w: *const c_void,
                                   convDesc: cudnnConvolutionDescriptor,
                                   algo: cudnnConvolutionFwdAlgo,
                                   workspace: *mut c_void,
                                   workSpaceSizeInBytes: size_t,
                                   beta: *const c_void,
                                   yDesc: cudnnTensorDescriptor,
                                   y: *mut c_void)
                                   -> cudnnStatus;
    pub fn cudnnGetConvolutionBackwardFilterAlgorithm(handle: cudnnHandle,
                                                      xDesc: cudnnTensorDescriptor,
                                                      dyDesc: cudnnTensorDescriptor,
                                                      convDesc: cudnnConvolutionDescriptor,
                                                      dwDesc: cudnnFilterDescriptor,
                                                      preference: cudnnConvolutionBwdFilterPreference,
                                                      memoryLimitInBytes: size_t,
                                                      algo: *mut cudnnConvolutionBwdFilterAlgo)
-> cudnnStatus;
    pub fn cudnnGetConvolutionBackwardFilterWorkspaceSize(handle: cudnnHandle,
                                                          xDesc: cudnnTensorDescriptor,
                                                          dyDesc: cudnnTensorDescriptor,
                                                          convDesc: cudnnConvolutionDescriptor,
                                                          dwDesc: cudnnFilterDescriptor,
                                                          algo: cudnnConvolutionBwdFilterAlgo,
                                                          sizeInBytes: *mut size_t)
                                                          -> cudnnStatus;
    pub fn cudnnConvolutionBackwardFilter(handle: cudnnHandle,
                                          alpha: *const c_void,
                                          xDesc: cudnnTensorDescriptor,
                                          x: *const c_void,
                                          dyDesc: cudnnTensorDescriptor,
                                          dy: *const c_void,
                                          convDesc: cudnnConvolutionDescriptor,
                                          algo: cudnnConvolutionBwdFilterAlgo,
                                          workSpace: *mut c_void,
                                          workSpaceSizeInBytes: size_t,
                                          beta: *const c_void,
                                          dwDesc: cudnnFilterDescriptor,
                                          dw: *mut c_void)
                                          -> cudnnStatus;
}
