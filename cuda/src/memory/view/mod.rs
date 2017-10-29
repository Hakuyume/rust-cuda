use std::marker;

use super::{Repr, ReprMut};

pub struct View<'a, T>
    where T: 'a
{
    ptr: *const T,
    len: usize,
    _lifetime: marker::PhantomData<&'a ()>,
}

pub unsafe fn from_raw_parts<'a, T>(ptr: *const T, len: usize) -> View<'a, T> {
    View {
        ptr,
        len,
        _lifetime: marker::PhantomData::default(),
    }
}

impl<'a, T> Repr<T> for View<'a, T>
    where T: 'a
{
    fn as_ptr(&self) -> *const T {
        self.ptr
    }
    fn len(&self) -> usize {
        self.len
    }
}

pub struct ViewMut<'a, T>
    where T: 'a
{
    ptr: *mut T,
    len: usize,
    _lifetime: marker::PhantomData<&'a mut ()>,
}

pub unsafe fn from_raw_parts_mut<'a, T>(ptr: *mut T, len: usize) -> ViewMut<'a, T> {
    ViewMut {
        ptr,
        len,
        _lifetime: marker::PhantomData::default(),
    }
}

impl<'a, T> Repr<T> for ViewMut<'a, T>
    where T: 'a
{
    fn as_ptr(&self) -> *const T {
        self.ptr
    }
    fn len(&self) -> usize {
        self.len
    }
}

impl<'a, T> ReprMut<T> for ViewMut<'a, T>
    where T: 'a
{
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

#[cfg(test)]
mod tests;
