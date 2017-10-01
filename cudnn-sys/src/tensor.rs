#![allow(non_camel_case_types)]

use {c_int, size_t};

use cudnnDataType;
use cudnnStatus;

pub enum cudnnTensorStruct {}
pub type cudnnTensorDescriptor = *mut cudnnTensorStruct;

#[link(name = "cudnn")]
extern "system" {
    pub fn cudnnCreateTensorDescriptor(tensorDesc: *mut cudnnTensorDescriptor) -> cudnnStatus;
    pub fn cudnnDestroyTensorDescriptor(tensorDesc: cudnnTensorDescriptor) -> cudnnStatus;
}

#[repr(C)]
pub enum cudnnTensorFormat {
    CUDNN_TENSOR_NCHW = 0,
    CUDNN_TENSOR_NHWC = 1,
    CUDNN_TENSOR_NCHW_VECT_C = 2,
}

#[link(name = "cudnn")]
extern "system" {
    pub fn cudnnSetTensor4dDescriptor(tensorDesc: cudnnTensorDescriptor,
                                      format: cudnnTensorFormat,
                                      dataType: cudnnDataType,
                                      n: c_int,
                                      c: c_int,
                                      h: c_int,
                                      w: c_int)
                                      -> cudnnStatus;
    pub fn cudnnSetTensor4dDescriptorEx(tensorDesc: cudnnTensorDescriptor,
                                        format: cudnnTensorFormat,
                                        dataType: cudnnDataType,
                                        n: c_int,
                                        c: c_int,
                                        h: c_int,
                                        w: c_int,
                                        nStride: c_int,
                                        cStride: c_int,
                                        hStride: c_int,
                                        wStride: c_int)
                                        -> cudnnStatus;
    pub fn cudnnGetTensorSizeInBytes(tensorDesc: cudnnTensorDescriptor,
                                     size: *mut size_t)
                                     -> cudnnStatus;
}
