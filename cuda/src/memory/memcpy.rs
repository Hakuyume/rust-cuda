use std::mem;

use cuda_sys;
use cuda_sys::c_void;

use Result;
use super::Slice;

pub trait MemcpyInto<D> {
    fn memcpy_into(&self, dst: D) -> Result<()>;
}

impl<'a, T> MemcpyInto<&'a mut Slice<T>> for [T] {
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

impl<'a, T> MemcpyInto<&'a mut [T]> for Slice<T> {
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

pub fn memcpy<D, S: ?Sized + MemcpyInto<D>>(dst: D, src: &S) -> Result<()> {
    src.memcpy_into(dst)
}
