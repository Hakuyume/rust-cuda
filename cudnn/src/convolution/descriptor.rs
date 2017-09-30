use std::marker;
use std::ptr;

use cudnn_sys;
use cudnn_sys::c_int;

use scalar;
use Result;

use super::Mode;

pub struct Descriptor<T: scalar::Scalar> {
    desc: cudnn_sys::cudnnConvolutionDescriptor,
    _dummy: marker::PhantomData<T>,
}

impl<T: scalar::Scalar> Descriptor<T> {
    fn new() -> Result<Descriptor<T>> {
        let mut desc = ptr::null_mut();
        unsafe { try_call!(cudnn_sys::cudnnCreateConvolutionDescriptor(&mut desc)) }
        Ok(Descriptor {
               desc,
               _dummy: marker::PhantomData::default(),
           })
    }

    pub fn new_2d(pad_h: usize,
                  pad_w: usize,
                  u: usize,
                  v: usize,
                  dilation_h: usize,
                  dilation_w: usize,
                  mode: Mode)
                  -> Result<Descriptor<T>> {
        let desc = try!(Descriptor::new());
        unsafe {
            try_call!(cudnn_sys::cudnnSetConvolution2dDescriptor(desc.as_raw(),
                                                                 pad_h as c_int,
                                                                 pad_w as c_int,
                                                                 u as c_int,
                                                                 v as c_int,
                                                                 dilation_h as c_int,
                                                                 dilation_w as c_int,
                                                                 mode.as_raw(),
                                                                 T::DATA_TYPE))
        };

        Ok(desc)
    }

    pub fn as_raw(&self) -> cudnn_sys::cudnnConvolutionDescriptor {
        self.desc
    }
}

impl<T: scalar::Scalar> Drop for Descriptor<T> {
    fn drop(&mut self) {
        unsafe { cudnn_sys::cudnnDestroyConvolutionDescriptor(self.desc) };
    }
}
