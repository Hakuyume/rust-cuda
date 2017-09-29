use std::mem;
use std::ptr;

use cuda_sys;
use cuda_sys::cudaError;
use cuda_sys::{c_void, size_t};

use Error;
use Result;

pub struct Memory<T> {
    ptr: *mut T,
    len: usize,
}

pub struct Slice<T> {
    _dummy: [T],
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

impl<T> Slice<T> {
    pub fn as_ptr(&self) -> *const T {
        self.repr().ptr as *const T
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.repr().ptr
    }

    pub fn len(&self) -> usize {
        self.repr().len
    }

    fn repr(&self) -> Repr<T> {
        unsafe { mem::transmute::<&Slice<T>, Repr<T>>(self) }
    }

    unsafe fn new<'a>(ptr: *mut T, len: usize) -> &'a Slice<T> {
        mem::transmute::<Repr<T>, &Slice<T>>(Repr { ptr, len })
    }

    unsafe fn new_mut<'a>(ptr: *mut T, len: usize) -> &'a mut Slice<T> {
        mem::transmute::<Repr<T>, &mut Slice<T>>(Repr { ptr, len })
    }
}

mod deref;
mod index;

pub fn copy_host_to_device<T>(dst: &mut Slice<T>, src: &[T]) -> Result<()> {
    assert_eq!(src.len(), dst.len());
    let error = unsafe {
        cuda_sys::cudaMemcpy(dst.as_mut_ptr() as *mut c_void,
                             src.as_ptr() as *const c_void,
                             mem::size_of::<T>() * src.len(),
                             cuda_sys::cudaMemcpyKind::cudaMemcpyHostToDevice)
    };
    match error {
        cudaError::cudaSuccess => Ok(()),
        e => Err(Error::from(e)),
    }
}

pub fn copy_device_to_host<T>(dst: &mut [T], src: &Slice<T>) -> Result<()> {
    assert_eq!(src.len(), dst.len());
    let error = unsafe {
        cuda_sys::cudaMemcpy(dst.as_mut_ptr() as *mut c_void,
                             src.as_ptr() as *const c_void,
                             mem::size_of::<T>() * src.len(),
                             cuda_sys::cudaMemcpyKind::cudaMemcpyDeviceToHost)
    };
    match error {
        cudaError::cudaSuccess => Ok(()),
        e => Err(Error::from(e)),
    }
}
