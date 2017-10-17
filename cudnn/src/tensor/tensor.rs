use cuda::slice;

use scalar;

use super::Descriptor;

pub struct Tensor<'a, T: 'a + scalar::Scalar> {
    desc: &'a Descriptor<T>,
    mem: &'a slice::Slice<T>,
}

impl<'a, T: scalar::Scalar> Tensor<'a, T> {
    pub fn new(desc: &'a Descriptor<T>, mem: &'a slice::Slice<T>) -> Tensor<'a, T> {
        assert_eq!(desc.len(), mem.len());
        Tensor { desc, mem }
    }

    pub fn desc(&self) -> &Descriptor<T> {
        self.desc
    }

    pub fn mem(&self) -> &slice::Slice<T> {
        self.mem
    }
}

pub struct TensorMut<'a, T: 'a + scalar::Scalar> {
    desc: &'a Descriptor<T>,
    mem: &'a mut slice::Slice<T>,
}

impl<'a, T: scalar::Scalar> TensorMut<'a, T> {
    pub fn new(desc: &'a Descriptor<T>, mem: &'a mut slice::Slice<T>) -> TensorMut<'a, T> {
        assert_eq!(desc.len(), mem.len());
        TensorMut { desc, mem }
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
