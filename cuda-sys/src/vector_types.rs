#![allow(non_camel_case_types)]

use c_uint;

#[repr(C)]
pub struct dim3 {
    pub x: c_uint,
    pub y: c_uint,
    pub z: c_uint,
}
