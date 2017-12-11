use std::mem;
use std::ptr;

use cuda_sys;
use cuda_sys::{c_void, size_t};

use Result;
use super::{Repr, ReprMut};

pub struct Memory<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> Memory<T> {
    pub fn new(len: usize) -> Result<Memory<T>> {
        let mut ptr = ptr::null_mut();
        unsafe { try_call!(cuda_sys::cudaMalloc(&mut ptr, (mem::size_of::<T>() * len) as size_t)) }
        Ok(Memory {
               ptr: ptr as *mut T,
               len,
           })
    }
}

impl<T> Drop for Memory<T> {
    fn drop(&mut self) {
        unsafe { cuda_sys::cudaFree(self.ptr as *mut c_void) }
    }
}

impl<T> Repr<T> for Memory<T> {
    fn as_ptr(&self) -> *const T {
        self.ptr
    }
    fn len(&self) -> usize {
        self.len
    }
}

impl<T> ReprMut<T> for Memory<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

#[cfg(test)]
mod tests;
