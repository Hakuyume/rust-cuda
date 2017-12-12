use std::mem;

use cuda::memory::Repr;

use Result;
use scalar;
use tensor;
use filter;

pub trait MemoryDescriptor<T> {
    fn get_size(&self) -> Result<usize>;
    fn check_memory<R>(&self, mem: &R) -> Result<()>
        where R: Repr<T>
    {
        assert_eq!(self.get_size()?, mem::size_of::<T>() * mem.len());
        Ok(())
    }
}

impl<T> MemoryDescriptor<T> for tensor::Descriptor<T>
    where T: scalar::Scalar
{
    fn get_size(&self) -> Result<usize> {
        self.get_size_in_bytes()
    }
}

impl<T> MemoryDescriptor<T> for filter::Descriptor<T>
    where T: scalar::Scalar
{
    fn get_size(&self) -> Result<usize> {
        let (_, k, c, h, w) = self.get_4d()?;
        Ok((mem::size_of::<T>() * k * c * h * w))
    }
}
