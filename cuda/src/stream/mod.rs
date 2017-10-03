use std::ptr;

use cuda_sys;

pub struct Stream {
    stream: Option<cuda_sys::cudaStream>,
}

impl Stream {
    pub fn default() -> Stream {
        Stream { stream: None }
    }

    pub fn as_ptr(&self) -> cuda_sys::cudaStream {
        self.stream.unwrap_or(ptr::null_mut())
    }

    pub fn as_mut_ptr(&mut self) -> cuda_sys::cudaStream {
        self.stream.unwrap_or(ptr::null_mut())
    }
}
