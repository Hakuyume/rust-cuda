use std::marker;
use std::ptr;

use cudnn_sys;
use cudnn_sys::c_int;

use scalar;
use Result;
use tensor;

pub struct FilterDescriptor<T: scalar::Scalar> {
    desc: cudnn_sys::cudnnFilterDescriptor,
    len: usize,
    _dummy: marker::PhantomData<T>,
}

impl<T: scalar::Scalar> FilterDescriptor<T> {
    fn new() -> Result<FilterDescriptor<T>> {
        let mut desc = ptr::null_mut();
        unsafe { try_call!(cudnn_sys::cudnnCreateFilterDescriptor(&mut desc)) }
        Ok(FilterDescriptor {
               desc,
               len: 0,
               _dummy: marker::PhantomData::default(),
           })
    }

    pub fn new_4d(format: tensor::Format,
                  k: usize,
                  c: usize,
                  h: usize,
                  w: usize)
                  -> Result<FilterDescriptor<T>> {
        let mut desc = try!(FilterDescriptor::new());
        unsafe {
            try_call!(cudnn_sys::cudnnSetFilter4dDescriptor(desc.as_raw(),
                                                            T::DATA_TYPE,
                                                            format.as_raw(),
                                                            k as c_int,
                                                            c as c_int,
                                                            h as c_int,
                                                            w as c_int))
        };
        desc.len = k * c * h * w;

        Ok(desc)
    }

    pub fn as_raw(&self) -> cudnn_sys::cudnnFilterDescriptor {
        self.desc
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl<T: scalar::Scalar> Drop for FilterDescriptor<T> {
    fn drop(&mut self) {
        unsafe { cudnn_sys::cudnnDestroyFilterDescriptor(self.desc) };
    }
}
