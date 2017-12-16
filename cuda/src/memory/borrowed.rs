use std::marker;

use super::{Ptr, PtrMut};

pub struct Borrowed<'a, T> {
    ptr: *const T,
    _lifetime: marker::PhantomData<&'a ()>,
}

impl<'a, T> Borrowed<'a, T> {
    pub unsafe fn from_raw(ptr: *const T) -> Borrowed<'a, T> {
        Borrowed {
            ptr,
            _lifetime: marker::PhantomData::default(),
        }
    }
}

impl<'a, T> Ptr for Borrowed<'a, T> {
    type Type = T;
    fn as_ptr(&self) -> *const T {
        self.ptr
    }
}

pub struct BorrowedMut<'a, T> {
    ptr: *mut T,
    _lifetime: marker::PhantomData<&'a mut ()>,
}

impl<'a, T> BorrowedMut<'a, T> {
    pub unsafe fn from_raw_mut(ptr: *mut T) -> BorrowedMut<'a, T> {
        BorrowedMut {
            ptr,
            _lifetime: marker::PhantomData::default(),
        }
    }
}

impl<'a, T> Ptr for BorrowedMut<'a, T> {
    type Type = T;
    fn as_ptr(&self) -> *const T {
        self.ptr
    }
}

impl<'a, T> PtrMut for BorrowedMut<'a, T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}
