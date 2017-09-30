#![allow(non_camel_case_types)]

use c_void;

use cudnnHandle;
use cudnnStatus;
use cudnnTensorDescriptor;

#[repr(C)]
pub enum cudnnSoftmaxAlgorithm {
    CUDNN_SOFTMAX_FAST = 0,
    CUDNN_SOFTMAX_ACCURATE = 1,
    CUDNN_SOFTMAX_LOG = 2,
}

#[repr(C)]
pub enum cudnnSoftmaxMode {
    CUDNN_SOFTMAX_MODE_INSTANCE = 0,
    CUDNN_SOFTMAX_MODE_CHANNEL = 1,
}

#[link(name = "cudnn")]
extern "system" {
    pub fn cudnnSoftmaxForward(handle: cudnnHandle,
                               algo: cudnnSoftmaxAlgorithm,
                               mode: cudnnSoftmaxMode,
                               alpha: *const c_void,
                               xDesc: *const cudnnTensorDescriptor,
                               x: *const c_void,
                               beta: *const c_void,
                               yDesc: *const cudnnTensorDescriptor,
                               y: *mut c_void)
                               -> cudnnStatus;
}
