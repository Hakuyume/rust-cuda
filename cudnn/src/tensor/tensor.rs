use std::marker;

use cuda::memory;

use scalar;

use super::Descriptor;

pub struct Tensor<'a, T: 'a + scalar::Scalar> {
    pub desc: &'a Descriptor<T>,
    pub mem: &'a memory::Slice<T>,
    _dummy: marker::PhantomData<T>,
}

pub struct TensorMut<'a, T: 'a + scalar::Scalar> {
    pub desc: &'a Descriptor<T>,
    pub mem: &'a mut memory::Slice<T>,
    _dummy: marker::PhantomData<T>,
}

impl<T: scalar::Scalar> Descriptor<T> {
    pub fn wrap<'a>(&'a self, mem: &'a memory::Slice<T>) -> Tensor<'a, T> {
        assert_eq!(self.len(), mem.len());
        Tensor {
            desc: self,
            mem,
            _dummy: marker::PhantomData::default(),
        }
    }

    pub fn wrap_mut<'a>(&'a self, mem: &'a mut memory::Slice<T>) -> TensorMut<'a, T> {
        assert_eq!(self.len(), mem.len());
        TensorMut {
            desc: self,
            mem,
            _dummy: marker::PhantomData::default(),
        }
    }
}
