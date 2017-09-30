use cuda;
use cuda::memory;

use scalar;

use super::Descriptor;
use super::{Filter, FilterMut};

pub struct OwnedFilter<T: scalar::Scalar> {
    desc: Descriptor<T>,
    mem: memory::Memory<T>,
}

impl<T: scalar::Scalar> OwnedFilter<T> {
    pub fn new(desc: Descriptor<T>) -> cuda::Result<OwnedFilter<T>> {
        let mem = try!(memory::Memory::new(desc.len()));
        Ok(OwnedFilter { desc, mem })
    }
}

impl<T: scalar::Scalar> Filter<T> for OwnedFilter<T> {
    fn desc(&self) -> &Descriptor<T> {
        &self.desc
    }
    fn mem(&self) -> &memory::Slice<T> {
        &self.mem
    }
}

impl<T: scalar::Scalar> FilterMut<T> for OwnedFilter<T> {
    fn mem_mut(&mut self) -> &mut memory::Slice<T> {
        &mut self.mem
    }
}
