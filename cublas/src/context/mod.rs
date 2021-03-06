use std::ptr;

use cublas_sys;

use Result;
use PointerMode;

pub struct Context {
    handle: cublas_sys::cublasHandle_t,
}

impl Context {
    pub fn new() -> Result<Context> {
        let mut handle = ptr::null_mut();
        unsafe { try_call!(cublas_sys::cublasCreate_v2(&mut handle)) };
        Ok(Context { handle })
    }

    pub fn as_mut_ptr(&mut self) -> cublas_sys::cublasHandle_t {
        self.handle
    }

    pub fn get_pointer_mode(&self) -> Result<PointerMode> {
        let mut mode = 0;
        unsafe { try_call!(cublas_sys::cublasGetPointerMode_v2(self.handle, &mut mode)) }
        Ok(mode.into())
    }

    pub fn set_pointer_mode(&mut self, mode: PointerMode) -> Result<()> {
        unsafe { try_call!(cublas_sys::cublasSetPointerMode_v2(self.handle, mode as _)) }
        Ok(())
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { cublas_sys::cublasDestroy_v2(self.handle) };
    }
}

#[cfg(test)]
mod tests;
