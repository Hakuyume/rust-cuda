use std::marker;
use std::os::raw::c_int;
use std::ptr;

use cudnn_sys;

use Result;
use scalar;
use tensor;

pub struct Descriptor<T>
    where T: scalar::Scalar
{
    desc: cudnn_sys::cudnnFilterDescriptor_t,
    _type: marker::PhantomData<T>,
}

impl<T> Descriptor<T>
    where T: scalar::Scalar
{
    pub fn new() -> Result<Descriptor<T>> {
        let mut desc = ptr::null_mut();
        unsafe { try_call!(cudnn_sys::cudnnCreateFilterDescriptor(&mut desc)) }
        Ok(Descriptor {
               desc,
               _type: marker::PhantomData::default(),
           })
    }

    pub fn as_ptr(&self) -> cudnn_sys::cudnnFilterDescriptor_t {
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
            try_call!(cudnn_sys::cudnnSetFilter4dDescriptor(self.desc,
                                                            T::DATA_TYPE,
                                                            format.into(),
                                                            k as c_int,
                                                            c as c_int,
                                                            h as c_int,
                                                            w as c_int))
        }
        Ok(())
    }

    pub fn get_4d(&self) -> Result<(tensor::Format, usize, usize, usize, usize)> {
        let mut data_type = T::DATA_TYPE;
        let mut format = cudnn_sys::cudnnTensorFormat_t_CUDNN_TENSOR_NCHW;
        let mut k = 0;
        let mut c = 0;
        let mut h = 0;
        let mut w = 0;
        unsafe {
            try_call!(cudnn_sys::cudnnGetFilter4dDescriptor(self.desc,
                                                            &mut data_type,
                                                            &mut format,
                                                            &mut k,
                                                            &mut c,
                                                            &mut h,
                                                            &mut w))
        }
        assert_eq!(data_type, T::DATA_TYPE);
        Ok((format.into(), k as usize, c as usize, h as usize, w as usize))
    }
}

impl<T> Drop for Descriptor<T>
    where T: scalar::Scalar
{
    fn drop(&mut self) {
        unsafe { cudnn_sys::cudnnDestroyFilterDescriptor(self.desc) };
    }
}
