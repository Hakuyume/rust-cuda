use std;
use cuda_sys;
use cuda_sys::{cudaError_t, c_void, size_t};

use Error;
use Result;

pub struct Memory<T> {
    ptr: *mut T,
    length: usize,
}

impl<T> Memory<T> {
    pub fn new(length: usize) -> Result<Memory<T>> {
        let mut ptr = std::ptr::null_mut::<c_void>();
        let error = unsafe {
            cuda_sys::cudaMalloc(&mut ptr, (std::mem::size_of::<T>() * length) as size_t)
        };
        match error {
            cudaError_t::cudaSuccess => {
                let ptr = ptr as *mut T;
                Ok(Memory { ptr, length })
            }
            e => Err(Error::from(e)),
        }
    }

    pub fn len(&self) -> usize {
        self.length
    }
}

impl<T> Drop for Memory<T> {
    fn drop(&mut self) {
        unsafe { cuda_sys::cudaFree(self.ptr as *mut c_void) };
    }
}
