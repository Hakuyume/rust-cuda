use std::mem;
use std::os::raw::c_void;
use std::ptr;

use cuda_sys;

use Result;

use super::{Ptr, PtrMut};

pub struct Owned<T> {
    ptr: *mut T,
}

impl<T> Owned<T> {
    pub fn new(len: usize) -> Result<Owned<T>> {
        let mut ptr = ptr::null_mut();
        unsafe { try_call!(cuda_sys::cudaMalloc(&mut ptr, mem::size_of::<T>() * len)) }
        Ok(Owned { ptr: ptr as *mut T })
    }
}

impl<T> Drop for Owned<T> {
    fn drop(&mut self) {
        unsafe { cuda_sys::cudaFree(self.ptr as *mut c_void) };
    }
}

impl<T> Ptr for Owned<T> {
    type Type = T;
    fn as_ptr(&self) -> *const T {
        self.ptr
    }
}

impl<T> PtrMut for Owned<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}
