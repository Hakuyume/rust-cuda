use cuda::memory;

use scalar;

use super::Descriptor;

pub trait Tensor<T: scalar::Scalar> {
    fn desc(&self) -> &Descriptor<T>;
    fn mem(&self) -> &memory::Slice<T>;
}

pub trait TensorMut<T: scalar::Scalar>: Tensor<T> {
    fn mem_mut(&mut self) -> &mut memory::Slice<T>;
}
