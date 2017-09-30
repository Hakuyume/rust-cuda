use std::marker;
use std::ptr;

use cudnn_sys;
use cudnn_sys::c_int;

use scalar;
use Result;

pub enum Mode {
    Convolution,
    CrossCorrelation,
}

pub struct ConvolutionDescriptor<T: scalar::Scalar> {
    desc: cudnn_sys::cudnnConvolutionDescriptor,
    _dummy: marker::PhantomData<T>,
}

impl<T: scalar::Scalar> ConvolutionDescriptor<T> {
    fn new() -> Result<ConvolutionDescriptor<T>> {
        let mut desc = ptr::null_mut();
        unsafe { try_call!(cudnn_sys::cudnnCreateConvolutionDescriptor(&mut desc)) }
        Ok(ConvolutionDescriptor {
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
                  -> Result<ConvolutionDescriptor<T>> {
        let desc = try!(ConvolutionDescriptor::new());

        let mode = match mode {
            Mode::Convolution => cudnn_sys::cudnnConvolutionMode::CUDNN_CONVOLUTION,
            Mode::CrossCorrelation => cudnn_sys::cudnnConvolutionMode::CUDNN_CROSS_CORRELATION,
        };
        unsafe {
            try_call!(cudnn_sys::cudnnSetConvolution2dDescriptor(desc.as_raw(),
                                                                 pad_h as c_int,
                                                                 pad_w as c_int,
                                                                 u as c_int,
                                                                 v as c_int,
                                                                 dilation_h as c_int,
                                                                 dilation_w as c_int,
                                                                 mode,
                                                                 T::DATA_TYPE))
        };

        Ok(desc)
    }

    pub fn as_raw(&self) -> cudnn_sys::cudnnConvolutionDescriptor {
        self.desc
    }
}

impl<T: scalar::Scalar> Drop for ConvolutionDescriptor<T> {
    fn drop(&mut self) {
        unsafe { cudnn_sys::cudnnDestroyConvolutionDescriptor(self.desc) };
    }
}
