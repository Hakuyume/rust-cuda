use std::marker;

use cuda::slice;

use scalar;

use super::Descriptor;

pub struct Tensor<'a, T: 'a + scalar::Scalar> {
    pub desc: &'a Descriptor<T>,
    pub mem: &'a slice::Slice<T>,
    _dummy: marker::PhantomData<T>,
}

impl<'a, T: scalar::Scalar> Tensor<'a, T> {
    pub fn new(desc: &'a Descriptor<T>, mem: &'a slice::Slice<T>) -> Tensor<'a, T> {
        assert_eq!(desc.len(), mem.len());
        Tensor {
            desc,
            mem,
            _dummy: marker::PhantomData::default(),
        }
    }
}

pub struct TensorMut<'a, T: 'a + scalar::Scalar> {
    pub desc: &'a Descriptor<T>,
    pub mem: &'a mut slice::Slice<T>,
    _dummy: marker::PhantomData<T>,
}

impl<'a, T: scalar::Scalar> TensorMut<'a, T> {
    pub fn new(desc: &'a Descriptor<T>, mem: &'a mut slice::Slice<T>) -> TensorMut<'a, T> {
        assert_eq!(desc.len(), mem.len());
        TensorMut {
            desc,
            mem,
            _dummy: marker::PhantomData::default(),
        }
    }
}
