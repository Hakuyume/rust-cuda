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

impl<T, D> MemcpyFrom<[T]> for D
    where D: ?Sized + ops::DerefMut<Target = Slice<T>>
{
    fn memcpy_from(&mut self, src: &[T]) -> Result<()> {
        let dst = self.deref_mut();
        assert_eq!(src.len(), dst.len());
        unsafe {
            try_call!(cuda_sys::cudaMemcpy(dst.as_mut_ptr() as *mut c_void,
                                           src.as_ptr() as *const c_void,
                                           mem::size_of::<T>() * src.len(),
                                           cuda_sys::cudaMemcpyKind::cudaMemcpyHostToDevice))
        }
        Ok(())
    }
}

impl<T, D> MemcpyFrom<Slice<T>> for D
    where D: ?Sized + ops::DerefMut<Target = [T]>
{
    fn memcpy_from(&mut self, src: &Slice<T>) -> Result<()> {
        let dst = self.deref_mut();

        assert_eq!(src.len(), dst.len());
        unsafe {
            try_call!(cuda_sys::cudaMemcpy(dst.as_mut_ptr() as *mut c_void,
                                           src.as_ptr() as *const c_void,
                                           mem::size_of::<T>() * src.len(),
                                           cuda_sys::cudaMemcpyKind::cudaMemcpyDeviceToHost))
        }
        Ok(())
    }
}
