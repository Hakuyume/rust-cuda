use std::mem;
use std::ops;

use cuda_sys;
use cuda_sys::c_void;

use Result;
use super::Slice;

pub trait MemcpyFrom<S: ?Sized> {
    fn memcpy_from(&mut self, src: &S) -> Result<()>;
}

pub fn memcpy<D, S>(dst: &mut D, src: &S) -> Result<()>
    where S: ?Sized,
          D: ?Sized + MemcpyFrom<S>
{
    dst.memcpy_from(src)
}

impl<T> MemcpyFrom<[T]> for Slice<T> {
    fn memcpy_from(&mut self, src: &[T]) -> Result<()> {
        assert_eq!(src.len(), self.len());
        unsafe {
            try_call!(cuda_sys::cudaMemcpy(self.as_mut_ptr() as *mut c_void,
                                           src.as_ptr() as *const c_void,
                                           mem::size_of::<T>() * self.len(),
                                           cuda_sys::cudaMemcpyKind::cudaMemcpyHostToDevice))
        }
        Ok(())
    }
}

impl<T> MemcpyFrom<Slice<T>> for [T] {
    fn memcpy_from(&mut self, src: &Slice<T>) -> Result<()> {
        assert_eq!(src.len(), self.len());
        unsafe {
            try_call!(cuda_sys::cudaMemcpy(self.as_mut_ptr() as *mut c_void,
                                           src.as_ptr() as *const c_void,
                                           mem::size_of::<T>() * self.len(),
                                           cuda_sys::cudaMemcpyKind::cudaMemcpyDeviceToHost))
        }
        Ok(())
    }
}

impl<T, D> MemcpyFrom<[T]> for D
    where D: ops::DerefMut<Target = Slice<T>>
{
    fn memcpy_from(&mut self, src: &[T]) -> Result<()> {
        self.deref_mut().memcpy_from(src)
    }
}
