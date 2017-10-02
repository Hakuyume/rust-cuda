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
    pub fn new(desc: &'a Descriptor<T>, mem: &'a slice::Slice<T>) -> Option<Tensor<'a, T>> {
        if desc.len() == mem.len() {
            Some(Tensor {
                     desc,
                     mem,
                     _dummy: marker::PhantomData::default(),
                 })
        } else {
            None
        }
    }
}

pub struct TensorMut<'a, T: 'a + scalar::Scalar> {
    pub desc: &'a Descriptor<T>,
    pub mem: &'a mut slice::Slice<T>,
    _dummy: marker::PhantomData<T>,
}

impl<'a, T: scalar::Scalar> TensorMut<'a, T> {
    pub fn new(desc: &'a Descriptor<T>, mem: &'a mut slice::Slice<T>) -> Option<TensorMut<'a, T>> {
        if desc.len() == mem.len() {
            Some(TensorMut {
                     desc,
                     mem,
                     _dummy: marker::PhantomData::default(),
                 })
        } else {
            None
        }
    }
}
