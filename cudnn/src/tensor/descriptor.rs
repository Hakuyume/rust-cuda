use std::marker;
use std::ptr;

use cudnn_sys;
use cudnn_sys::c_int;

use scalar;
use Result;

use super::Format;

pub struct Descriptor<T: scalar::Scalar> {
    desc: cudnn_sys::cudnnTensorDescriptor,
    len: usize,
    _dummy: marker::PhantomData<T>,
}

impl<T: scalar::Scalar> Descriptor<T> {
    fn new() -> Result<Descriptor<T>> {
        let mut desc = ptr::null_mut();
        unsafe { try_call!(cudnn_sys::cudnnCreateTensorDescriptor(&mut desc)) }
        Ok(Descriptor {
               desc,
               len: 0,
               _dummy: marker::PhantomData::default(),
           })
    }

    pub fn new_4d(format: Format, n: usize, c: usize, h: usize, w: usize) -> Result<Descriptor<T>> {
        let mut desc = try!(Descriptor::new());
        unsafe {
            try_call!(cudnn_sys::cudnnSetTensor4dDescriptor(desc.as_raw(),
                                                            format.as_raw(),
                                                            T::DATA_TYPE,
                                                            n as c_int,
                                                            c as c_int,
                                                            h as c_int,
                                                            w as c_int))
        };
        desc.len = n * c * h * w;

        Ok(desc)
    }

    pub fn as_raw(&self) -> cudnn_sys::cudnnTensorDescriptor {
        self.desc
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl<T: scalar::Scalar> Drop for Descriptor<T> {
    fn drop(&mut self) {
        unsafe { cudnn_sys::cudnnDestroyTensorDescriptor(self.desc) };
    }
}
