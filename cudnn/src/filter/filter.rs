use cuda::slice;

use scalar;

use super::Descriptor;

pub struct Filter<'a, T: 'a + scalar::Scalar> {
    desc: &'a Descriptor<T>,
    mem: &'a slice::Slice<T>,
}

impl<'a, T: scalar::Scalar> Filter<'a, T> {
    pub fn new(desc: &'a Descriptor<T>, mem: &'a slice::Slice<T>) -> Filter<'a, T> {
        assert_eq!(desc.len(), mem.len());
        Filter { desc, mem }
    }

    pub fn desc(&self) -> &Descriptor<T> {
        self.desc
    }

    pub fn mem(&self) -> &slice::Slice<T> {
        self.mem
    }
}

pub struct FilterMut<'a, T: 'a + scalar::Scalar> {
    desc: &'a Descriptor<T>,
    mem: &'a mut slice::Slice<T>,
}

impl<'a, T: scalar::Scalar> FilterMut<'a, T> {
    pub fn new(desc: &'a Descriptor<T>, mem: &'a mut slice::Slice<T>) -> FilterMut<'a, T> {
        assert_eq!(desc.len(), mem.len());
        FilterMut { desc, mem }
    }

    pub fn desc(&self) -> &Descriptor<T> {
        self.desc
    }

    pub fn mem(&self) -> &slice::Slice<T> {
        self.mem
    }

    pub fn mem_mut(&mut self) -> &mut slice::Slice<T> {
        self.mem
    }
}
