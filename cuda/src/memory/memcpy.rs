use std::mem;
use std::ops;
use std::os::raw::c_void;

use cuda_sys;

use Result;

use super::{Ptr, PtrMut};
use super::Array;

pub trait MemcpyFrom<S>
    where S: ?Sized
{
    fn memcpy_from(&mut self, src: &S) -> Result<()>;
}

pub fn memcpy<D, S>(dst: &mut D, src: &S) -> Result<()>
    where D: ?Sized + MemcpyFrom<S>,
          S: ?Sized
{
    dst.memcpy_from(src)
}

impl<T, P> MemcpyFrom<[T]> for Array<P>
    where P: PtrMut<Type = T>
{
    fn memcpy_from(&mut self, src: &[T]) -> Result<()> {
        assert_eq!(src.len(), self.len());
        unsafe {
            try_call!(cuda_sys::cudaMemcpy(self.as_mut_ptr() as *mut c_void,
                                           src.as_ptr() as *const c_void,
                                           mem::size_of::<T>() * src.len(),
                                           cuda_sys::cudaMemcpyHostToDevice))
        }
        Ok(())
    }
}

impl<T, P, D> MemcpyFrom<Array<P>> for D
    where P: Ptr<Type = T>,
          D: ?Sized + ops::DerefMut<Target = [T]>
{
    fn memcpy_from(&mut self, src: &Array<P>) -> Result<()> {
        let dst = self.deref_mut();
        assert_eq!(src.len(), dst.len());
        unsafe {
            try_call!(cuda_sys::cudaMemcpy(dst.as_mut_ptr() as *mut c_void,
                                           src.as_ptr() as *const c_void,
                                           mem::size_of::<T>() * src.len(),
                                           cuda_sys::cudaMemcpyDeviceToHost))
        }
        Ok(())
    }
}
