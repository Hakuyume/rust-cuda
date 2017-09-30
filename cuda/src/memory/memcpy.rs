use std::mem;

use cuda_sys;
use cuda_sys::c_void;

use Result;
use super::Slice;

pub trait MemcpyFrom<S: ?Sized> {
    fn memcpy_from(self, src: &S) -> Result<()>;
}

pub fn memcpy<S: ?Sized, D: MemcpyFrom<S>>(dst: D, src: &S) -> Result<()> {
    dst.memcpy_from(src)
}

impl<'a, T> MemcpyFrom<[T]> for &'a mut Slice<T> {
    fn memcpy_from(self, src: &[T]) -> Result<()> {
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

impl<'a, T> MemcpyFrom<Slice<T>> for &'a mut [T] {
    fn memcpy_from(self, src: &Slice<T>) -> Result<()> {
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
