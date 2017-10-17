#![allow(non_camel_case_types)]

use c_void;

use cudnnStatus;
use cudnnHandle;
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
                               xDesc: cudnnTensorDescriptor,
                               x: *const c_void,
                               beta: *const c_void,
                               yDesc: cudnnTensorDescriptor,
                               y: *mut c_void)
                               -> cudnnStatus;
    pub fn cudnnSoftmaxBackward(handle: cudnnHandle,
                                algorithm: cudnnSoftmaxAlgorithm,
                                mode: cudnnSoftmaxMode,
                                alpha: *const c_void,
                                yDesc: cudnnTensorDescriptor,
                                yData: *const c_void,
                                dyDesc: cudnnTensorDescriptor,
                                dy: *const c_void,
                                beta: *const c_void,
                                dxDesc: cudnnTensorDescriptor,
                                dx: *mut c_void)
                                -> cudnnStatus;
}
