#![allow(non_camel_case_types)]

use c_int;

use cudnnDataType;
use cudnnStatus;
use cudnnTensorFormat;

pub enum cudnnFilterStruct {}
pub type cudnnFilterDescriptor = *mut cudnnFilterStruct;

#[link(name = "cudnn")]
extern "system" {
    pub fn cudnnCreateFilterDescriptor(filterDesc: *mut cudnnFilterDescriptor) -> cudnnStatus;
    pub fn cudnnSetFilter4dDescriptor(filterDesc: cudnnFilterDescriptor,
                                      dataType: cudnnDataType,
                                      format: cudnnTensorFormat,
                                      k: c_int,
                                      c: c_int,
                                      h: c_int,
                                      w: c_int)
                                      -> cudnnStatus;
    pub fn cudnnGetFilter4dDescriptor(filterDesc: cudnnFilterDescriptor,
                                      dataType: *mut cudnnDataType,
                                      format: *mut cudnnTensorFormat,
                                      k: *mut c_int,
                                      c: *mut c_int,
                                      h: *mut c_int,
                                      w: *mut c_int)
                                      -> cudnnStatus;
    pub fn cudnnDestroyFilterDescriptor(filterDesc: cudnnFilterDescriptor) -> cudnnStatus;
}
