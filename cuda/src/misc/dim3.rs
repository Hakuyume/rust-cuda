use std::os::raw::c_uint;

use cuda_sys;

pub struct Dim3 {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl Into<cuda_sys::dim3> for Dim3 {
    fn into(self) -> cuda_sys::dim3 {
        cuda_sys::dim3 {
            x: self.x as c_uint,
            y: self.y as c_uint,
            z: self.z as c_uint,
        }
    }
}
