use std::marker;

use cuda::memory;

use scalar;

use super::Descriptor;

pub struct Filter<'a, T: 'a + scalar::Scalar> {
    pub desc: &'a Descriptor<T>,
    pub mem: &'a memory::Slice<T>,
    _dummy: marker::PhantomData<T>,
}

impl<'a, T: scalar::Scalar> Filter<'a, T> {
    pub fn new(desc: &'a Descriptor<T>, mem: &'a memory::Slice<T>) -> Option<Filter<'a, T>> {
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
    pub mem: &'a mut memory::Slice<T>,
    _dummy: marker::PhantomData<T>,
}

impl<'a, T: scalar::Scalar> FilterMut<'a, T> {
    pub fn new(desc: &'a Descriptor<T>, mem: &'a mut memory::Slice<T>) -> Option<FilterMut<'a, T>> {
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
