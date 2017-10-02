#![allow(non_camel_case_types)]

use {c_double, c_void};

use cudnnNanPropagation;
use cudnnStatus;
use cudnnHandle;
use cudnnTensorDescriptor;

pub enum cudnnActivationStruct {}
pub type cudnnActivationDescriptor = *mut cudnnActivationStruct;

#[link(name = "cudnn")]
extern "system" {
    pub fn cudnnCreateActivationDescriptor(activationDesc: *mut cudnnActivationDescriptor)
                                           -> cudnnStatus;
    pub fn cudnnDestroyActivationDescriptor(activationDesc: cudnnActivationDescriptor)
                                            -> cudnnStatus;
}

#[repr(C)]
pub enum cudnnActivationMode {
    CUDNN_ACTIVATION_SIGMOID = 0,
    CUDNN_ACTIVATION_RELU = 1,
    CUDNN_ACTIVATION_TANH = 2,
    CUDNN_ACTIVATION_CLIPPED_RELU = 3,
    CUDNN_ACTIVATION_ELU = 4,
}

#[link(name = "cudnn")]
extern "system" {
    pub fn cudnnSetActivationDescriptor(activationDesc: cudnnActivationDescriptor,
                                        mode: cudnnActivationMode,
                                        reluNanOpt: cudnnNanPropagation,
                                        coef: c_double)
                                        -> cudnnStatus;
}

#[link(name = "cudnn")]
extern "system" {
    pub fn cudnnActivationForward(handle: cudnnHandle,
                                  activationDesc: cudnnActivationDescriptor,
                                  alpha: *const c_void,
                                  srcDesc: cudnnTensorDescriptor,
                                  srcData: *const c_void,
                                  beta: *const c_void,
                                  destDesc: cudnnTensorDescriptor,
                                  destData: *mut c_void)
                                  -> cudnnStatus;
}
