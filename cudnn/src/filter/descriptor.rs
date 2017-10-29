use std::marker;
use std::ptr;

use cudnn_sys;
use cudnn_sys::c_int;

use scalar;
use Result;
use tensor;

pub struct Descriptor<T>
    where T: scalar::Scalar
{
    desc: cudnn_sys::cudnnFilterDescriptor,
    len: usize,
    _dummy: marker::PhantomData<T>,
}

impl<T> Descriptor<T>
    where T: scalar::Scalar
{
    pub fn new() -> Result<Descriptor<T>> {
        let mut desc = ptr::null_mut();
        unsafe { try_call!(cudnn_sys::cudnnCreateFilterDescriptor(&mut desc)) }
        Ok(Descriptor {
               desc,
               len: 0,
               _dummy: marker::PhantomData::default(),
           })
    }

    pub fn as_ptr(&self) -> cudnn_sys::cudnnFilterDescriptor {
        self.desc
    }

    pub fn as_mut_ptr(&mut self) -> cudnn_sys::cudnnFilterDescriptor {
        self.desc
    }

    pub fn set_4d(&mut self,
                  format: tensor::Format,
                  k: usize,
                  c: usize,
                  h: usize,
                  w: usize)
                  -> Result<()> {
        unsafe {
            try_call!(cudnn_sys::cudnnSetFilter4dDescriptor(self.as_mut_ptr(),
                                                            T::DATA_TYPE,
                                                            format.into(),
                                                            k as c_int,
                                                            c as c_int,
                                                            h as c_int,
                                                            w as c_int))
        }
        self.len = k * c * h * w;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl<T> Drop for Descriptor<T>
    where T: scalar::Scalar
{
    fn drop(&mut self) {
        unsafe { cudnn_sys::cudnnDestroyFilterDescriptor(self.desc) };
    }
}
