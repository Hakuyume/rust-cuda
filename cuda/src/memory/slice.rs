use std::marker;

use super::{View, ViewMut};

pub struct Slice<'a, T>
    where T: 'a
{
    ptr: *const T,
    len: usize,
    _lifetime: marker::PhantomData<&'a ()>,
}

pub unsafe fn from_raw_parts<'a, T>(ptr: *const T, len: usize) -> Slice<'a, T> {
    Slice {
        ptr,
        len,
        _lifetime: marker::PhantomData::default(),
    }
}

impl<'a, T> View<T> for Slice<'a, T>
    where T: 'a
{
    fn as_ptr(&self) -> *const T {
        self.ptr
    }
    fn len(&self) -> usize {
        self.len
    }
}

pub struct SliceMut<'a, T>
    where T: 'a
{
    ptr: *mut T,
    len: usize,
    _lifetime: marker::PhantomData<&'a mut ()>,
}

pub unsafe fn from_raw_parts_mut<'a, T>(ptr: *mut T, len: usize) -> SliceMut<'a, T> {
    SliceMut {
        ptr,
        len,
        _lifetime: marker::PhantomData::default(),
    }
}

impl<'a, T> View<T> for SliceMut<'a, T>
    where T: 'a
{
    fn as_ptr(&self) -> *const T {
        self.ptr
    }
    fn len(&self) -> usize {
        self.len
    }
}

impl<'a, T> ViewMut<T> for SliceMut<'a, T>
    where T: 'a
{
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}
