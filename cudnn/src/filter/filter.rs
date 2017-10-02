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
    pub fn new(desc: &'a Descriptor<T>, mem: &'a slice::Slice<T>) -> Filter<'a, T> {
        assert_eq!(desc.len(), mem.len());
        Filter {
            desc,
            mem,
            _dummy: marker::PhantomData::default(),
        }
    }
}

pub struct FilterMut<'a, T: 'a + scalar::Scalar> {
    pub desc: &'a Descriptor<T>,
    pub mem: &'a mut slice::Slice<T>,
    _dummy: marker::PhantomData<T>,
}

impl<'a, T: scalar::Scalar> FilterMut<'a, T> {
    pub fn new(desc: &'a Descriptor<T>, mem: &'a mut slice::Slice<T>) -> FilterMut<'a, T> {
        assert_eq!(desc.len(), mem.len());
        FilterMut {
            desc,
            mem,
            _dummy: marker::PhantomData::default(),
        }
    }
}
