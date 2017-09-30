#![allow(non_camel_case_types)]

use cudnnStatus;

pub enum cudnnContext {}
pub type cudnnHandle = *mut cudnnContext;

#[link(name = "cudnn")]
extern "system" {
    pub fn cudnnCreate(handle: *mut cudnnHandle) -> cudnnStatus;
    pub fn cudnnDestroy(handle: cudnnHandle) -> cudnnStatus;
}
