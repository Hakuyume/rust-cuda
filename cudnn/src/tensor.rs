use std::ptr;

use cuda;

use cudnn_sys;
use cudnn_sys::c_int;

use scalar::Scalar;
use Result;

struct TensorDescriptor {
    desc: cudnn_sys::cudnnTensorDescriptor,
}

impl TensorDescriptor {
    fn new() -> Result<TensorDescriptor> {
        let mut desc = ptr::null_mut();
        unsafe { try_call!(cudnn_sys::cudnnCreateTensorDescriptor(&mut desc)) }
        Ok(TensorDescriptor { desc })
    }
}

impl Drop for TensorDescriptor {
    fn drop(&mut self) {
        unsafe { cudnn_sys::cudnnDestroyTensorDescriptor(self.desc) };
    }
}

pub struct Tensor<'a, T: 'a + Scalar> {
    mem: &'a mut cuda::memory::Slice<T>,
    desc: TensorDescriptor,
}

impl<'a, T: Scalar> Tensor<'a, T> {
    pub fn new_4d(mem: &'a mut cuda::memory::Slice<T>,
                  n: usize,
                  c: usize,
                  h: usize,
                  w: usize)
                  -> Result<Tensor<'a, T>> {
        assert_eq!(mem.len(), n * c * h * w);
        let desc = try!(TensorDescriptor::new());
        unsafe {
            try_call!(cudnn_sys::cudnnSetTensor4dDescriptor(desc.desc,
                                                            cudnn_sys::cudnnTensorFormat::CUDNN_TENSOR_NCHW,
                                                            T::DATA_TYPE,
                                                            n as c_int,c as c_int,h as c_int,w as c_int))
        };
        Ok(Tensor { mem, desc })
    }

    pub fn as_ptr(&self) -> *const T {
        self.mem.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.mem.as_mut_ptr()
    }

    pub fn desc(&self) -> cudnn_sys::cudnnTensorDescriptor {
        self.desc.desc
    }
}
