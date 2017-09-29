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
}

impl<T> Drop for Memory<T> {
    fn drop(&mut self) {
        unsafe { cuda_sys::cudaFree(self.ptr as *mut c_void) };
    }
}

impl<T> ops::Deref for Memory<T> {
    type Target = Slice<T>;
    fn deref(&self) -> &Slice<T> {
        unsafe { Slice::new(self.ptr, self.len) }
    }
}

impl<T> ops::DerefMut for Memory<T> {
    fn deref_mut(&mut self) -> &mut Slice<T> {
        unsafe { Slice::new_mut(self.ptr, self.len) }
    }
}

impl<T> Slice<T> {
    unsafe fn new<'a>(ptr: *mut T, len: usize) -> &'a Slice<T> {
        mem::transmute::<Repr<T>, &Slice<T>>(Repr { ptr, len })
    }

    unsafe fn new_mut<'a>(ptr: *mut T, len: usize) -> &'a mut Slice<T> {
        mem::transmute::<Repr<T>, &mut Slice<T>>(Repr { ptr, len })
    }

    pub fn ptr(&self) -> *const T {
        unsafe { mem::transmute::<&Slice<T>, Repr<T>>(self).ptr as *const T }
    }

    pub fn ptr_mut(&mut self) -> *mut T {
        unsafe { mem::transmute::<&Slice<T>, Repr<T>>(self).ptr }
    }

    pub fn len(&self) -> usize {
        unsafe { mem::transmute::<&Slice<T>, Repr<T>>(self).len }
    }

    fn get_slice(&self, start: Option<usize>, end: Option<usize>) -> &Slice<T> {
        unsafe {
            let repr = mem::transmute::<&Slice<T>, Repr<T>>(self);
            let start = start.unwrap_or(0);
            let end = end.unwrap_or(repr.len);
            assert!(start < end);
            assert!(end <= repr.len);
            Slice::new(repr.ptr.offset(start as isize), end - start)
        }
    }

    fn get_slice_mut(&mut self, start: Option<usize>, end: Option<usize>) -> &mut Slice<T> {
        unsafe {
            let repr = mem::transmute::<&Slice<T>, Repr<T>>(self);
            let start = start.unwrap_or(0);
            let end = end.unwrap_or(repr.len);
            assert!(start < end);
            assert!(end <= repr.len);
            Slice::new_mut(repr.ptr.offset(start as isize), end - start)
        }
    }
}

impl<T> ops::Index<ops::RangeFull> for Slice<T> {
    type Output = Slice<T>;
    fn index(&self, _: ops::RangeFull) -> &Slice<T> {
        self.get_slice(None, None)
    }
}

impl<T> ops::Index<ops::Range<usize>> for Slice<T> {
    type Output = Slice<T>;
    fn index(&self, index: ops::Range<usize>) -> &Slice<T> {
        self.get_slice(Some(index.start), Some(index.end))
    }
}

impl<T> ops::Index<ops::RangeFrom<usize>> for Slice<T> {
    type Output = Slice<T>;
    fn index(&self, index: ops::RangeFrom<usize>) -> &Slice<T> {
        self.get_slice(Some(index.start), None)
    }
}

impl<T> ops::Index<ops::RangeTo<usize>> for Slice<T> {
    type Output = Slice<T>;
    fn index(&self, index: ops::RangeTo<usize>) -> &Slice<T> {
        self.get_slice(None, Some(index.end))
    }
}

impl<T> ops::IndexMut<ops::RangeFull> for Slice<T> {
    fn index_mut(&mut self, _: ops::RangeFull) -> &mut Slice<T> {
        self.get_slice_mut(None, None)
    }
}

impl<T> ops::IndexMut<ops::Range<usize>> for Slice<T> {
    fn index_mut(&mut self, index: ops::Range<usize>) -> &mut Slice<T> {
        self.get_slice_mut(Some(index.start), Some(index.end))
    }
}

impl<T> ops::IndexMut<ops::RangeFrom<usize>> for Slice<T> {
    fn index_mut(&mut self, index: ops::RangeFrom<usize>) -> &mut Slice<T> {
        self.get_slice_mut(Some(index.start), None)
    }
}

impl<T> ops::IndexMut<ops::RangeTo<usize>> for Slice<T> {
    fn index_mut(&mut self, index: ops::RangeTo<usize>) -> &mut Slice<T> {
        self.get_slice_mut(None, Some(index.end))
    }
}
