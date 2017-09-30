use std::ptr;

use cudnn_sys;

pub struct Context {
    handle: cudnn_sys::cudnnHandle,
}

impl Context {
    pub fn new() -> Context {
        let mut handle = ptr::null_mut();
        unsafe { cudnn_sys::cudnnCreate(&mut handle) };
        Context { handle }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { cudnn_sys::cudnnDestroy(self.handle) };
    }
}
