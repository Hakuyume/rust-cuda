use std::ptr;

use cudnn_sys;

use Result;

pub struct Context {
    handle: cudnn_sys::cudnnHandle,
}

impl Context {
    pub fn new() -> Result<Context> {
        let mut handle = ptr::null_mut();
        unsafe { try_call!(cudnn_sys::cudnnCreate(&mut handle)) };
        Ok(Context { handle })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { cudnn_sys::cudnnDestroy(self.handle) };
    }
}
