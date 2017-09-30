use std::marker;

use cuda::memory;

use scalar;

use super::Descriptor;

pub struct Filter<'a, T: 'a + scalar::Scalar> {
    pub desc: &'a Descriptor<T>,
    pub mem: &'a memory::Slice<T>,
    _dummy: marker::PhantomData<T>,
}

pub struct FilterMut<'a, T: 'a + scalar::Scalar> {
    pub desc: &'a Descriptor<T>,
    pub mem: &'a mut memory::Slice<T>,
    _dummy: marker::PhantomData<T>,
}

impl<T: scalar::Scalar> Descriptor<T> {
    pub fn wrap<'a>(&'a self, mem: &'a memory::Slice<T>) -> Filter<'a, T> {
        assert_eq!(self.len(), mem.len());
        Filter {
            desc: self,
            mem,
            _dummy: marker::PhantomData::default(),
        }
    }

    pub fn wrap_mut<'a>(&'a self, mem: &'a mut memory::Slice<T>) -> FilterMut<'a, T> {
        assert_eq!(self.len(), mem.len());
        FilterMut {
            desc: self,
            mem,
            _dummy: marker::PhantomData::default(),
        }
    }
}
