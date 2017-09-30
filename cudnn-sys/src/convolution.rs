#![allow(non_camel_case_types)]

use c_int;

use cudnnDataType;
use cudnnStatus;

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
