use std::marker;

use cuda::slice;

use scalar;

use super::Descriptor;

pub struct Filter<'a, T: 'a + scalar::Scalar> {
    pub desc: &'a Descriptor<T>,
    pub mem: &'a slice::Slice<T>,
    _dummy: marker::PhantomData<T>,
}

impl<'a, T: scalar::Scalar> Filter<'a, T> {
    pub fn new(desc: &'a Descriptor<T>, mem: &'a slice::Slice<T>) -> Option<Filter<'a, T>> {
        if desc.len() == mem.len() {
            Some(Filter {
                     desc,
                     mem,
                     _dummy: marker::PhantomData::default(),
                 })
        } else {
            None
        }
    }
}

pub struct FilterMut<'a, T: 'a + scalar::Scalar> {
    pub desc: &'a Descriptor<T>,
    pub mem: &'a mut slice::Slice<T>,
    _dummy: marker::PhantomData<T>,
}

impl<'a, T: scalar::Scalar> FilterMut<'a, T> {
    pub fn new(desc: &'a Descriptor<T>, mem: &'a mut slice::Slice<T>) -> Option<FilterMut<'a, T>> {
        if desc.len() == mem.len() {
            Some(FilterMut {
                     desc,
                     mem,
                     _dummy: marker::PhantomData::default(),
                 })
        } else {
            None
        }
    }
}
