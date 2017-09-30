use cuda::memory;

use scalar;

use super::Descriptor;
use super::{Filter, FilterMut};

pub struct BorrowedFilter<'a, T: 'a + scalar::Scalar> {
    desc: &'a Descriptor<T>,
    mem: &'a memory::Slice<T>,
}

impl<'a, T: scalar::Scalar> BorrowedFilter<'a, T> {
    pub fn new(desc: &'a Descriptor<T>, mem: &'a memory::Slice<T>) -> BorrowedFilter<'a, T> {
        assert_eq!(desc.len(), mem.len());
        BorrowedFilter { desc, mem }
    }
}

impl<'a, T: scalar::Scalar> Filter<T> for BorrowedFilter<'a, T> {
    fn desc(&self) -> &Descriptor<T> {
        self.desc
    }
    fn mem(&self) -> &memory::Slice<T> {
        self.mem
    }
}

pub struct BorrowedFilterMut<'a, T: 'a + scalar::Scalar> {
    desc: &'a Descriptor<T>,
    mem: &'a mut memory::Slice<T>,
}

impl<'a, T: scalar::Scalar> BorrowedFilterMut<'a, T> {
    pub fn new(desc: &'a Descriptor<T>, mem: &'a mut memory::Slice<T>) -> BorrowedFilterMut<'a, T> {
        assert_eq!(desc.len(), mem.len());
        BorrowedFilterMut { desc, mem }
    }
}

impl<'a, T: scalar::Scalar> Filter<T> for BorrowedFilterMut<'a, T> {
    fn desc(&self) -> &Descriptor<T> {
        self.desc
    }
    fn mem(&self) -> &memory::Slice<T> {
        self.mem
    }
}

impl<'a, T: scalar::Scalar> FilterMut<T> for BorrowedFilterMut<'a, T> {
    fn mem_mut(&mut self) -> &mut memory::Slice<T> {
        self.mem
    }
}
