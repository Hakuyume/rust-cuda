use cuda::memory;

use scalar;

use super::TensorDescriptor;
use super::{Tensor, TensorMut};

pub struct BorrowedTensor<'a, T: 'a + scalar::Scalar> {
    desc: &'a TensorDescriptor<T>,
    mem: &'a memory::Slice<T>,
}

impl<'a, T: scalar::Scalar> BorrowedTensor<'a, T> {
    pub fn new(desc: &'a TensorDescriptor<T>, mem: &'a memory::Slice<T>) -> BorrowedTensor<'a, T> {
        assert_eq!(desc.len(), mem.len());
        BorrowedTensor { desc, mem }
    }
}

impl<'a, T: scalar::Scalar> Tensor<T> for BorrowedTensor<'a, T> {
    fn desc(&self) -> &TensorDescriptor<T> {
        self.desc
    }
    fn mem(&self) -> &memory::Slice<T> {
        self.mem
    }
}

pub struct BorrowedTensorMut<'a, T: 'a + scalar::Scalar> {
    desc: &'a TensorDescriptor<T>,
    mem: &'a mut memory::Slice<T>,
}

impl<'a, T: scalar::Scalar> BorrowedTensorMut<'a, T> {
    pub fn new(desc: &'a TensorDescriptor<T>,
               mem: &'a mut memory::Slice<T>)
               -> BorrowedTensorMut<'a, T> {
        assert_eq!(desc.len(), mem.len());
        BorrowedTensorMut { desc, mem }
    }
}

impl<'a, T: scalar::Scalar> Tensor<T> for BorrowedTensorMut<'a, T> {
    fn desc(&self) -> &TensorDescriptor<T> {
        self.desc
    }
    fn mem(&self) -> &memory::Slice<T> {
        self.mem
    }
}

impl<'a, T: scalar::Scalar> TensorMut<T> for BorrowedTensorMut<'a, T> {
    fn mem_mut(&mut self) -> &mut memory::Slice<T> {
        self.mem
    }
}
