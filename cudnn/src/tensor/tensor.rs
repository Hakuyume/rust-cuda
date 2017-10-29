use cuda::memory;

use scalar;

use super::Descriptor;

pub struct Tensor<'a, T>
    where T: 'a + scalar::Scalar
{
    desc: &'a Descriptor<T>,
    mem: memory::View<'a, T>,
}

impl<'a, T> Tensor<'a, T>
    where T: scalar::Scalar
{
    pub fn new<R>(desc: &'a Descriptor<T>, mem: &'a R) -> Tensor<'a, T>
        where R: memory::Repr<T>
    {
        assert_eq!(desc.len(), mem.len());
        Tensor {
            desc,
            mem: mem.view(),
        }
    }

    pub fn desc(&self) -> &Descriptor<T> {
        self.desc
    }

    pub fn mem(&self) -> &memory::View<'a, T> {
        &self.mem
    }
}

pub struct TensorMut<'a, T>
    where T: 'a + scalar::Scalar
{
    desc: &'a Descriptor<T>,
    mem: memory::ViewMut<'a, T>,
}

impl<'a, T> TensorMut<'a, T>
    where T: 'a + scalar::Scalar
{
    pub fn new<R>(desc: &'a Descriptor<T>, mem: &'a mut R) -> TensorMut<'a, T>
        where R: memory::ReprMut<T>
    {
        assert_eq!(desc.len(), mem.len());
        TensorMut {
            desc,
            mem: mem.view_mut(),
        }
    }

    pub fn desc(&self) -> &Descriptor<T> {
        self.desc
    }

    pub fn mem(&self) -> &memory::ViewMut<'a, T> {
        &self.mem
    }

    pub fn mem_mut(&mut self) -> &mut memory::ViewMut<'a, T> {
        &mut self.mem
    }
}
