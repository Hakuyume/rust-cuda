use std::mem;

use cuda_sys;
use cuda_sys::c_void;

use Result;
use super::Slice;

pub trait MemcpyFrom<S: ?Sized> {
    fn memcpy_from(&mut self, src: &S) -> Result<()>;
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