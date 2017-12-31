use std::mem;
use std::os::raw::c_void;
use std::ptr;

use cuda_sys;

use Result;

use super::{Repr, ReprMut};

pub struct OwnedRepr<T> {
    ptr: *mut T,
}

impl<T> OwnedRepr<T> {
    pub fn new(len: usize) -> Result<OwnedRepr<T>> {
        let mut ptr = ptr::null_mut();
        unsafe { try_call!(cuda_sys::cudaMalloc(&mut ptr, mem::size_of::<T>() * len)) }
        Ok(OwnedRepr { ptr: ptr as *mut T })
    }
}

impl<T> Drop for OwnedRepr<T> {
    fn drop(&mut self) {
        unsafe { cuda_sys::cudaFree(self.ptr as *mut c_void) };
    }
}

impl<T> Repr for OwnedRepr<T> {
    type Type = T;
    fn as_ptr(&self) -> *const T {
        self.ptr
    }
}

impl<T> ReprMut for OwnedRepr<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}
