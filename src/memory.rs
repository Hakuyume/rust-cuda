use std::mem;
use std::ops;
use std::ptr;

use cuda_sys;
use cuda_sys::{cudaError, c_void, size_t};

use Error;
use Result;

pub struct Memory<T> {
    ptr: *mut T,
    len: usize,
}

pub struct Slice<T> {
    _stub: [T],
}

#[repr(C)]
struct Repr<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> Memory<T> {
    pub fn new(len: usize) -> Result<Memory<T>> {
        let mut ptr = ptr::null_mut::<c_void>();
        let error =
            unsafe { cuda_sys::cudaMalloc(&mut ptr, (mem::size_of::<T>() * len) as size_t) };
        match error {
            cudaError::cudaSuccess => {
                Ok(Memory {
                       ptr: ptr as *mut T,
                       len,
                   })
            }
            e => Err(Error::from(e)),
        }
    }

    fn get_slice(&self, offset: usize, len: usize) -> &Slice<T> {
        assert!(offset + len <= self.len);
        unsafe {
            let repr = Repr {
                ptr: self.ptr.offset(offset as isize),
                len,
            };
            mem::transmute::<Repr<T>, &Slice<T>>(repr)
        }
    }

    fn get_slice_mut(&mut self, offset: usize, len: usize) -> &mut Slice<T> {
        assert!(offset + len <= self.len);
        unsafe {
            let repr = Repr {
                ptr: self.ptr.offset(offset as isize),
                len,
            };
            mem::transmute::<Repr<T>, &mut Slice<T>>(repr)
        }
    }
}

impl<T> Drop for Memory<T> {
    fn drop(&mut self) {
        unsafe { cuda_sys::cudaFree(self.ptr as *mut c_void) };
    }
}

impl<T> ops::Deref for Memory<T> {
    type Target = Slice<T>;
    fn deref(&self) -> &Slice<T> {
        self.get_slice(0, self.len)
    }
}

impl<T> ops::DerefMut for Memory<T> {
    fn deref_mut(&mut self) -> &mut Slice<T> {
        let len = self.len;
        self.get_slice_mut(0, len)
    }
}

impl<T> ops::Index<ops::RangeFull> for Memory<T> {
    type Output = Slice<T>;
    fn index(&self, _: ops::RangeFull) -> &Slice<T> {
        self.get_slice(0, self.len)
    }
}

impl<T> Slice<T> {
    pub fn len(&self) -> usize {
        unsafe { mem::transmute::<&Slice<T>, Repr<T>>(self).len }
    }
}
