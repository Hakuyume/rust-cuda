use std::mem;
use std::ops;
use std::ptr;

use cuda_sys;
use cuda_sys::{c_void, size_t};

use Result;
use slice;

pub struct Memory<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> Memory<T> {
    pub fn new(len: usize) -> Result<Memory<T>> {
        let mut ptr = ptr::null_mut::<c_void>();
        unsafe { try_call!(cuda_sys::cudaMalloc(&mut ptr, (mem::size_of::<T>() * len) as size_t)) }
        Ok(Memory {
               ptr: ptr as *mut T,
               len,
           })
    }
}

impl<T> Drop for Memory<T> {
    fn drop(&mut self) {
        unsafe { cuda_sys::cudaFree(self.ptr as *mut c_void) };
    }
}

impl<T> ops::Deref for Memory<T> {
    type Target = slice::Slice<T>;
    fn deref(&self) -> &slice::Slice<T> {
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T> ops::DerefMut for Memory<T> {
    fn deref_mut(&mut self) -> &mut slice::Slice<T> {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}
