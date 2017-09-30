use cuda;
use cuda::memory;

use scalar;

use super::Descriptor;
use super::{Tensor, TensorMut};

pub struct OwnedTensor<T: scalar::Scalar> {
    desc: Descriptor<T>,
    mem: memory::Memory<T>,
}

impl<T: scalar::Scalar> OwnedTensor<T> {
    pub fn new(desc: Descriptor<T>) -> cuda::Result<OwnedTensor<T>> {
        let mem = try!(memory::Memory::new(desc.len()));
        Ok(OwnedTensor { desc, mem })
    }
}

impl<T: scalar::Scalar> Tensor<T> for OwnedTensor<T> {
    fn desc(&self) -> &Descriptor<T> {
        &self.desc
    }
    fn mem(&self) -> &memory::Slice<T> {
        &self.mem
    }
}

impl<T: scalar::Scalar> TensorMut<T> for OwnedTensor<T> {
    fn mem_mut(&mut self) -> &mut memory::Slice<T> {
        &mut self.mem
    }
}
