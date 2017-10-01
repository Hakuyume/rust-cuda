use std::marker;
use std::mem;
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
    pub fn new() -> Result<Descriptor<T>> {
        let mut desc = ptr::null_mut();
        unsafe { try_call!(cudnn_sys::cudnnCreateTensorDescriptor(&mut desc)) }
        Ok(Descriptor {
               desc,
               len: 0,
               _dummy: marker::PhantomData::default(),
           })
    }

    pub fn as_raw(&self) -> cudnn_sys::cudnnTensorDescriptor {
        self.desc
    }

    pub fn get_size(&self) -> Result<usize> {
        let mut size = 0;
        unsafe { try_call!(cudnn_sys::cudnnGetTensorSizeInBytes(self.as_raw(), &mut size)) }
        Ok(size as usize)
    }

    pub fn set_4d(&mut self, format: Format, n: usize, c: usize, h: usize, w: usize) -> Result<()> {
        unsafe {
            try_call!(cudnn_sys::cudnnSetTensor4dDescriptor(self.as_raw(),
                                                            format.as_raw(),
                                                            T::DATA_TYPE,
                                                            n as c_int,
                                                            c as c_int,
                                                            h as c_int,
                                                            w as c_int))
        }
        let size = try!(self.get_size());
        assert_eq!(size % mem::size_of::<T>(), 0);
        self.len = size / mem::size_of::<T>();
        Ok(())
    }

    pub fn set_4d_ex(&mut self,
                     n: usize,
                     c: usize,
                     h: usize,
                     w: usize,
                     n_stride: usize,
                     c_stride: usize,
                     h_stride: usize,
                     w_stride: usize)
                     -> Result<()> {
        unsafe {
            try_call!(cudnn_sys::cudnnSetTensor4dDescriptorEx(self.as_raw(),
                                                              T::DATA_TYPE,
                                                              n as c_int,
                                                              c as c_int,
                                                              h as c_int,
                                                              w as c_int,
                                                              n_stride as c_int,
                                                              c_stride as c_int,
                                                              h_stride as c_int,
                                                              w_stride as c_int))
        }
        let size = try!(self.get_size());
        assert_eq!(size % mem::size_of::<T>(), 0);
        self.len = size / mem::size_of::<T>();
        Ok(())
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
