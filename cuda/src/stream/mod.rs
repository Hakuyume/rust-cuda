use std::ptr;

use cuda_sys;

use Result;

pub struct Stream {
    ptr: cuda_sys::cudaStream_t,
}

impl Stream {
    pub fn new() -> Result<Stream> {
        let mut ptr = ptr::null_mut();
        unsafe { try_call!(cuda_sys::cudaStreamCreate(&mut ptr)) }
        Ok(Stream { ptr })
    }

    pub fn as_ptr(&self) -> cuda_sys::cudaStream_t {
        self.ptr
    }

    pub fn synchronize(&self) -> Result<()> {
        unsafe { try_call!(cuda_sys::cudaStreamSynchronize(self.ptr)) }
        Ok(())
    }
}

impl Default for Stream {
    fn default() -> Stream {
        Stream { ptr: ptr::null_mut() }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { cuda_sys::cudaStreamDestroy(self.ptr) };
        }
    }
}

#[cfg(test)]
mod tests;
