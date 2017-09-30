#![allow(non_camel_case_types)]

use c_int;

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
}
