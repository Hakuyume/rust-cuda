use std::mem;

use cuda_sys;
use cuda_sys::c_void;

use Result;
use super::Slice;

pub trait MemcpyInto<D: ?Sized> {
    fn memcpy_into(&self, dst: &mut D) -> Result<()>;
}

impl<T> MemcpyInto<Slice<T>> for [T] {
    fn memcpy_into(&self, dst: &mut Slice<T>) -> Result<()> {
        assert_eq!(self.len(), dst.len());
        unsafe {
            try_call!(cuda_sys::cudaMemcpy(dst.as_mut_ptr() as *mut c_void,
                                           self.as_ptr() as *const c_void,
                                           mem::size_of::<T>() * self.len(),
                                           cuda_sys::cudaMemcpyKind::cudaMemcpyHostToDevice))
        }
        Ok(())
    }
}

impl<T> MemcpyInto<[T]> for Slice<T> {
    fn memcpy_into(&self, dst: &mut [T]) -> Result<()> {
        assert_eq!(self.len(), dst.len());
        unsafe {
            try_call!(cuda_sys::cudaMemcpy(dst.as_mut_ptr() as *mut c_void,
                                           self.as_ptr() as *const c_void,
                                           mem::size_of::<T>() * self.len(),
                                           cuda_sys::cudaMemcpyKind::cudaMemcpyDeviceToHost))
        }
        Ok(())
    }
}

pub fn memcpy<D: ?Sized, S: ?Sized + MemcpyInto<D>>(dst: &mut D, src: &S) -> Result<()> {
    src.memcpy_into(dst)
}
