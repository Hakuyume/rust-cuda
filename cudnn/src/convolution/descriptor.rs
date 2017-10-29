use std::marker;
use std::ptr;

use cudnn_sys;
use cudnn_sys::c_int;

use scalar;
use Result;

use super::Mode;

pub struct Descriptor<T>
    where T: scalar::Scalar
{
    desc: cudnn_sys::cudnnConvolutionDescriptor,
    _dummy: marker::PhantomData<T>,
}

impl<T> Descriptor<T>
    where T: scalar::Scalar
{
    pub fn new() -> Result<Descriptor<T>> {
        let mut desc = ptr::null_mut();
        unsafe { try_call!(cudnn_sys::cudnnCreateConvolutionDescriptor(&mut desc)) }
        Ok(Descriptor {
               desc,
               _dummy: marker::PhantomData::default(),
           })
    }

    pub fn as_ptr(&self) -> cudnn_sys::cudnnConvolutionDescriptor {
        self.desc
    }

    pub fn as_mut_ptr(&mut self) -> cudnn_sys::cudnnConvolutionDescriptor {
        self.desc
    }

    pub fn set_2d(&mut self,
                  pad_h: usize,
                  pad_w: usize,
                  u: usize,
                  v: usize,
                  dilation_h: usize,
                  dilation_w: usize,
                  mode: Mode)
                  -> Result<()> {
        unsafe {
            try_call!(cudnn_sys::cudnnSetConvolution2dDescriptor(self.as_mut_ptr(),
                                                                 pad_h as c_int,
                                                                 pad_w as c_int,
                                                                 u as c_int,
                                                                 v as c_int,
                                                                 dilation_h as c_int,
                                                                 dilation_w as c_int,
                                                                 mode.into(),
                                                                 T::DATA_TYPE))
        }
        Ok(())
    }
}

impl<T> Drop for Descriptor<T>
    where T: scalar::Scalar
{
    fn drop(&mut self) {
        unsafe { cudnn_sys::cudnnDestroyConvolutionDescriptor(self.desc) };
    }
}
