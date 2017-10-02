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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Get4DOutput {
    pub n: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub n_stride: usize,
    pub c_stride: usize,
    pub h_stride: usize,
    pub w_stride: usize,
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

    pub fn as_ptr(&self) -> cudnn_sys::cudnnTensorDescriptor {
        self.desc
    }

    pub fn as_mut_ptr(&mut self) -> cudnn_sys::cudnnTensorDescriptor {
        self.desc
    }

    pub fn get_size(&self) -> Result<usize> {
        let mut size = 0;
        unsafe { try_call!(cudnn_sys::cudnnGetTensorSizeInBytes(self.as_ptr(), &mut size)) }
        Ok(size as usize)
    }

    pub fn set_4d(&mut self, format: Format, n: usize, c: usize, h: usize, w: usize) -> Result<()> {
        unsafe {
            try_call!(cudnn_sys::cudnnSetTensor4dDescriptor(self.as_mut_ptr(),
                                                            format.into(),
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

    pub fn get_4d(&self) -> Result<Get4DOutput> {
        let mut data_type = T::DATA_TYPE;
        let mut n = 0;
        let mut c = 0;
        let mut h = 0;
        let mut w = 0;
        let mut n_stride = 0;
        let mut c_stride = 0;
        let mut h_stride = 0;
        let mut w_stride = 0;
        unsafe {
            try_call!(cudnn_sys::cudnnGetTensor4dDescriptor(self.as_ptr(),
                                                            &mut data_type,
                                                            &mut n,
                                                            &mut c,
                                                            &mut h,
                                                            &mut w,
                                                            &mut n_stride,
                                                            &mut c_stride,
                                                            &mut h_stride,
                                                            &mut w_stride))
        }
        assert_eq!(data_type, T::DATA_TYPE);
        Ok(Get4DOutput {
               n: n as usize,
               c: c as usize,
               h: h as usize,
               w: w as usize,
               n_stride: n_stride as usize,
               c_stride: c_stride as usize,
               h_stride: h_stride as usize,
               w_stride: w_stride as usize,
           })
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
            try_call!(cudnn_sys::cudnnSetTensor4dDescriptorEx(self.as_mut_ptr(),
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
