use std::marker;
use std::os::raw::c_int;
use std::ptr;

use cudnn_sys;

use scalar;
use Result;

use super::Mode;

pub struct Descriptor<T>
    where T: scalar::Scalar
{
    desc: cudnn_sys::cudnnConvolutionDescriptor_t,
    _type: marker::PhantomData<T>,
}

impl<T> Descriptor<T>
    where T: scalar::Scalar
{
    pub fn new() -> Result<Descriptor<T>> {
        let mut desc = ptr::null_mut();
        unsafe { try_call!(cudnn_sys::cudnnCreateConvolutionDescriptor(&mut desc)) }
        Ok(Descriptor {
               desc,
               _type: marker::PhantomData::default(),
           })
    }

    pub fn as_ptr(&self) -> cudnn_sys::cudnnConvolutionDescriptor_t {
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
            try_call!(cudnn_sys::cudnnSetConvolution2dDescriptor(self.desc,
                                                                 pad_h as c_int,
                                                                 pad_w as c_int,
                                                                 u as c_int,
                                                                 v as c_int,
                                                                 dilation_h as c_int,
                                                                 dilation_w as c_int,
                                                                 mode as _,
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
