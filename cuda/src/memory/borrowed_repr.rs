use std::marker;

use super::{Repr, ReprMut};

pub struct BorrowedRepr<'a, T> {
    ptr: *const T,
    _lifetime: marker::PhantomData<&'a ()>,
}

impl<'a, T> BorrowedRepr<'a, T> {
    pub unsafe fn from_raw(ptr: *const T) -> BorrowedRepr<'a, T> {
        BorrowedRepr {
            ptr,
            _lifetime: marker::PhantomData,
        }
    }
}

impl<'a, T> Repr for BorrowedRepr<'a, T> {
    type Type = T;
    fn as_ptr(&self) -> *const T {
        self.ptr
    }
}

pub struct BorrowedReprMut<'a, T> {
    ptr: *mut T,
    _lifetime: marker::PhantomData<&'a mut ()>,
}

impl<'a, T> BorrowedReprMut<'a, T> {
    pub unsafe fn from_raw_mut(ptr: *mut T) -> BorrowedReprMut<'a, T> {
        BorrowedReprMut {
            ptr,
            _lifetime: marker::PhantomData,
        }
    }
}

impl<'a, T> Repr for BorrowedReprMut<'a, T> {
    type Type = T;
    fn as_ptr(&self) -> *const T {
        self.ptr
    }
}

impl<'a, T> ReprMut for BorrowedReprMut<'a, T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}
