use cuda::memory;

use scalar;

use super::Descriptor;

pub trait Filter<T: scalar::Scalar> {
    fn desc(&self) -> &Descriptor<T>;
    fn mem(&self) -> &memory::Slice<T>;
}

pub trait FilterMut<T: scalar::Scalar>: Filter<T> {
    fn mem_mut(&mut self) -> &mut memory::Slice<T>;
}
