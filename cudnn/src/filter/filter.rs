use cuda::memory;

use scalar;

use super::Descriptor;

pub struct Filter<'a, T>
    where T: 'a + scalar::Scalar
{
    desc: &'a Descriptor<T>,
    mem: memory::View<'a, T>,
}

impl<'a, T> Filter<'a, T>
    where T: scalar::Scalar
{
    pub fn new<R>(desc: &'a Descriptor<T>, mem: &'a R) -> Filter<'a, T>
        where R: memory::Repr<T>
    {
        assert_eq!(desc.len(), mem.len());
        Filter {
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

pub struct FilterMut<'a, T>
    where T: 'a + scalar::Scalar
{
    desc: &'a Descriptor<T>,
    mem: memory::ViewMut<'a, T>,
}

impl<'a, T> FilterMut<'a, T>
    where T: scalar::Scalar
{
    pub fn new<R>(desc: &'a Descriptor<T>, mem: &'a mut R) -> FilterMut<'a, T>
        where R: memory::ReprMut<T>
    {
        assert_eq!(desc.len(), mem.len());
        FilterMut {
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
