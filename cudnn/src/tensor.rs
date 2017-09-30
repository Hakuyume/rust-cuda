use std::marker;
use std::ptr;

use cudnn_sys;
use cudnn_sys::c_int;

use scalar;
use Result;

pub enum Format {
    NCHW,
    NHWC,
}

pub struct TensorDescriptor<T> {
    desc: cudnn_sys::cudnnTensorDescriptor,
    len: usize,
    _dummy: marker::PhantomData<T>,
}

impl<T: scalar::Scalar> TensorDescriptor<T> {
    fn new() -> Result<TensorDescriptor<T>> {
        let mut desc = ptr::null_mut();
        unsafe { try_call!(cudnn_sys::cudnnCreateTensorDescriptor(&mut desc)) }
        Ok(TensorDescriptor {
               desc,
               len: 0,
               _dummy: marker::PhantomData::default(),
           })
    }

    pub fn new_4d(format: Format,
                  n: usize,
                  c: usize,
                  h: usize,
                  w: usize)
                  -> Result<TensorDescriptor<T>> {
        let mut desc = try!(TensorDescriptor::new());

        let format = match format {
            Format::NCHW => cudnn_sys::cudnnTensorFormat::CUDNN_TENSOR_NCHW,
            Format::NHWC => cudnn_sys::cudnnTensorFormat::CUDNN_TENSOR_NHWC,
        };
        unsafe {
            try_call!(cudnn_sys::cudnnSetTensor4dDescriptor(desc.desc,
                                                            format,
                                                            T::DATA_TYPE,
                                                            n as c_int,
                                                            c as c_int,
                                                            h as c_int,
                                                            w as c_int))
        };
        desc.len = n * c * h * w;

        Ok(desc)
    }

    pub fn desc(&self) -> cudnn_sys::cudnnTensorDescriptor {
        self.desc
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl<T> Drop for TensorDescriptor<T> {
    fn drop(&mut self) {
        unsafe { cudnn_sys::cudnnDestroyTensorDescriptor(self.desc) };
    }
}
