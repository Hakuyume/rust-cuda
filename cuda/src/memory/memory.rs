use std::cell;
use std::mem;
use std::ptr;

use cuda_sys;
use cuda_sys::{c_void, size_t};

use Result;
use super::{View, ViewMut};

thread_local! {
    static MALLOC_HOOK: cell::RefCell<Option<Box<Fn(*const (), usize)>>> = cell::RefCell::new(None);
    static FREE_HOOK: cell::RefCell<Option<Box<Fn(*const (), usize)>>> = cell::RefCell::new(None);
}

pub struct Memory<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> Memory<T> {
    pub fn new(len: usize) -> Result<Memory<T>> {
        let mut ptr = ptr::null_mut::<c_void>();
        unsafe { try_call!(cuda_sys::cudaMalloc(&mut ptr, (mem::size_of::<T>() * len) as size_t)) }
        MALLOC_HOOK.with(|hook| if let Some(ref hook) = *hook.borrow() {
                             hook(ptr as *const (), mem::size_of::<T>() * len);
                         });
        Ok(Memory {
               ptr: ptr as *mut T,
               len,
           })
    }
}

impl<T> Drop for Memory<T> {
    fn drop(&mut self) {
        unsafe { cuda_sys::cudaFree(self.ptr as *mut c_void) };
        FREE_HOOK.with(|hook| if let Some(ref hook) = *hook.borrow() {
                           hook(self.ptr as *const (), mem::size_of::<T>() * self.len);
                       });
    }
}

impl<T> View<T> for Memory<T> {
    fn as_ptr(&self) -> *const T {
        self.ptr
    }
    fn len(&self) -> usize {
        self.len
    }
}

impl<T> ViewMut<T> for Memory<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

pub fn set_malloc_hook<F: 'static + Fn(*const (), usize)>(f: F) {
    MALLOC_HOOK.with(|hook| *hook.borrow_mut() = Some(Box::new(f)));
}

pub fn set_free_hook<F: 'static + Fn(*const (), usize)>(f: F) {
    FREE_HOOK.with(|hook| *hook.borrow_mut() = Some(Box::new(f)));
}
