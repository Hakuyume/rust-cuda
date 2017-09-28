use std::mem::size_of;
use std::ptr::null_mut;

use cuda_sys;
use cuda_sys::{cudaError, c_void, size_t};

use Error;
use Result;

pub struct Memory<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> Memory<T> {
    pub fn new(len: usize) -> Result<Memory<T>> {
        let mut ptr = null_mut::<c_void>();
        let error = unsafe { cuda_sys::cudaMalloc(&mut ptr, (size_of::<T>() * len) as size_t) };
        match error {
            cudaError::cudaSuccess => {
                Ok(Memory {
                       ptr: ptr as *mut T,
                       len,
                   })
            }
            e => Err(Error::from(e)),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl<T> Drop for Memory<T> {
    fn drop(&mut self) {
        unsafe { cuda_sys::cudaFree(self.ptr as *mut c_void) };
    }
}
