use std::os::raw::c_double;
use std::ptr;

use cudnn_sys;

use Result;

use super::Mode;

pub struct Descriptor {
    desc: cudnn_sys::cudnnActivationDescriptor_t,
}

impl Descriptor {
    pub fn new() -> Result<Descriptor> {
        let mut desc = ptr::null_mut();
        unsafe { try_call!(cudnn_sys::cudnnCreateActivationDescriptor(&mut desc)) }
        Ok(Descriptor { desc })
    }

    pub fn as_ptr(&self) -> cudnn_sys::cudnnActivationDescriptor_t {
        self.desc
    }

    pub fn set(&mut self, mode: Mode, relu_nan_opt: bool, coef: f64) -> Result<()> {
        let relu_nan_opt = if relu_nan_opt {
            cudnn_sys::CUDNN_PROPAGATE_NAN
        } else {
            cudnn_sys::CUDNN_NOT_PROPAGATE_NAN
        };
        unsafe {
            try_call!(cudnn_sys::cudnnSetActivationDescriptor(self.desc,
                                                              mode.into(),
                                                              relu_nan_opt,
                                                              coef as c_double))
        }
        Ok(())
    }
}

impl Drop for Descriptor {
    fn drop(&mut self) {
        unsafe { cudnn_sys::cudnnDestroyActivationDescriptor(self.desc) };
    }
}
