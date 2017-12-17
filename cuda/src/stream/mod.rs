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

    fn synchronize(&self) -> Result<()> {
        unsafe { try_call!(cuda_sys::cudaStreamSynchronize(self.stream)) }
        Ok(())
    }

    pub fn with<'a, F>(&'a self, f: F) -> SyncHandle<'a>
        where F: 'a + FnOnce(&Handle)
    {
        f(&Handle { stream: self.stream });
        SyncHandle { stream: Some(self) }
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
    stream: Option<&'a Stream>,
}

impl<'a> SyncHandle<'a> {
    pub fn synchronize(mut self) -> Result<()> {
        if let Some(stream) = self.stream.take() {
            stream.synchronize()?;
        }
        Ok(())
    }
}

impl<'a> Drop for SyncHandle<'a> {
    fn drop(&mut self) {
        if let Some(stream) = self.stream.take() {
            let _ = stream.synchronize();
        }
    }
}

#[cfg(test)]
mod tests;
