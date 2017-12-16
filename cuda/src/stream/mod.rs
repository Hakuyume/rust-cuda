use std::ptr;

use cuda_sys;

use Result;

pub struct Stream {
    stream: cuda_sys::cudaStream_t,
}

impl Stream {
    pub fn new() -> Result<Stream> {
        let mut stream = ptr::null_mut();
        unsafe { try_call!(cuda_sys::cudaStreamCreate(&mut stream)) }
        Ok(Stream { stream })
    }

    pub fn synchronize(&self) -> Result<()> {
        unsafe { try_call!(cuda_sys::cudaStreamSynchronize(self.stream)) }
        Ok(())
    }

    pub fn with<'a, F, T>(&'a self, f: F) -> (T, SyncHandle<'a>)
        where F: 'a + FnOnce(&Handle) -> T
    {
        (f(&Handle { stream: self.stream }), SyncHandle { stream: self })
    }
}

impl Default for Stream {
    fn default() -> Stream {
        Stream { stream: ptr::null_mut() }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe { cuda_sys::cudaStreamDestroy(self.stream) };
        }
    }
}

pub struct Handle {
    stream: cuda_sys::cudaStream_t,
}

impl Handle {
    pub fn as_ptr(&self) -> cuda_sys::cudaStream_t {
        self.stream
    }
}

pub struct SyncHandle<'a> {
    stream: &'a Stream,
}

impl<'a> Drop for SyncHandle<'a> {
    fn drop(&mut self) {
        self.stream.synchronize().unwrap()
    }
}

#[cfg(test)]
mod tests;
