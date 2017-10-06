use std::cell;
use std::mem;
use std::ops;
use std::ptr;

use cuda_sys;
use cuda_sys::{c_void, size_t};

use Result;
use slice;

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

impl<T> ops::Deref for Memory<T> {
    type Target = slice::Slice<T>;
    fn deref(&self) -> &slice::Slice<T> {
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T> ops::DerefMut for Memory<T> {
    fn deref_mut(&mut self) -> &mut slice::Slice<T> {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

pub fn set_malloc_hook<F: 'static + Fn(*const (), usize)>(f: F) {
    MALLOC_HOOK.with(|hook| *hook.borrow_mut() = Some(Box::new(f)));
}

pub fn set_free_hook<F: 'static + Fn(*const (), usize)>(f: F) {
    FREE_HOOK.with(|hook| *hook.borrow_mut() = Some(Box::new(f)));
}
