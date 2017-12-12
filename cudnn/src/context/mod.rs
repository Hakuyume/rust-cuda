use std::ptr;

use cudnn_sys;

use Result;

pub struct Context {
    handle: cudnn_sys::cudnnHandle_t,
}

impl Context {
    pub fn new() -> Result<Context> {
        let mut handle = ptr::null_mut();
        unsafe { try_call!(cudnn_sys::cudnnCreate(&mut handle)) };
        Ok(Context { handle })
    }

    pub fn as_mut_ptr(&mut self) -> cudnn_sys::cudnnHandle_t {
        self.handle
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { cudnn_sys::cudnnDestroy(self.handle) };
    }
}

#[cfg(test)]
mod tests;
