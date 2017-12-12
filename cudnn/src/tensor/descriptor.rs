use std::marker;
use std::os::raw::c_int;
use std::ptr;

use cudnn_sys;

use Result;
use scalar;

use super::Format;

pub struct Descriptor<T>
    where T: scalar::Scalar
{
    desc: cudnn_sys::cudnnTensorDescriptor_t,
    _type: marker::PhantomData<T>,
}

impl<T> Descriptor<T>
    where T: scalar::Scalar
{
    pub fn new() -> Result<Descriptor<T>> {
        let mut desc = ptr::null_mut();
        unsafe { try_call!(cudnn_sys::cudnnCreateTensorDescriptor(&mut desc)) }
        Ok(Descriptor {
               desc,
               _type: marker::PhantomData::default(),
           })
    }

    pub fn as_ptr(&self) -> cudnn_sys::cudnnTensorDescriptor_t {
        self.desc
    }

    pub fn set_4d(&mut self, format: Format, n: usize, c: usize, h: usize, w: usize) -> Result<()> {
        unsafe {
            try_call!(cudnn_sys::cudnnSetTensor4dDescriptor(self.desc,
                                                            format.into(),
                                                            T::DATA_TYPE,
                                                            n as c_int,
                                                            c as c_int,
                                                            h as c_int,
                                                            w as c_int))
        }
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
            try_call!(cudnn_sys::cudnnSetTensor4dDescriptorEx(self.desc,
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
        Ok(())
    }

    pub fn get_4d(&self) -> Result<(usize, usize, usize, usize, usize, usize, usize, usize)> {
        let mut data_type = 0;
        let mut n = 0;
        let mut c = 0;
        let mut h = 0;
        let mut w = 0;
        let mut n_stride = 0;
        let mut c_stride = 0;
        let mut h_stride = 0;
        let mut w_stride = 0;
        unsafe {
            try_call!(cudnn_sys::cudnnGetTensor4dDescriptor(self.desc,
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
        Ok((n as usize,
            c as usize,
            h as usize,
            w as usize,
            n_stride as usize,
            c_stride as usize,
            h_stride as usize,
            w_stride as usize))
    }

    pub fn get_size_in_bytes(&self) -> Result<usize> {
        let mut size_in_bytes = 0;
        unsafe {
            try_call!(cudnn_sys::cudnnGetTensorSizeInBytes(self.as_ptr(), &mut size_in_bytes))
        }
        Ok(size_in_bytes as usize)
    }
}

impl<T> Drop for Descriptor<T>
    where T: scalar::Scalar
{
    fn drop(&mut self) {
        unsafe { cudnn_sys::cudnnDestroyTensorDescriptor(self.desc) };
    }
}
