use std::collections::Bound;

use Result;
use nightly::collections::range;

use super::{Ptr, PtrMut};
use super::Owned;
use super::{Borrowed, BorrowedMut};

pub struct Array<P>
    where P: Ptr
{
    ptr: P,
    len: usize,
}

impl<T, P> Array<P>
    where P: Ptr<Type = T>
{
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn slice<R>(&self, range: R) -> Array<Borrowed<T>>
        where R: range::RangeArgument<usize>
    {
        let (start, end) = check_range(self.len, range);
        unsafe { Array::from_raw_parts(self.as_ptr().offset(start as isize), end - start) }
    }

    pub fn split_at(&self, mid: usize) -> (Array<Borrowed<T>>, Array<Borrowed<T>>) {
        let mid = check_mid(self.len(), mid);
        unsafe {
            (Array::from_raw_parts(self.as_ptr(), mid),
             Array::from_raw_parts(self.as_ptr().offset(mid as isize), self.len() - mid))
        }
    }
}

impl<T, P> Array<P>
    where P: PtrMut<Type = T>
{
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_mut_ptr()
    }

    pub fn slice_mut<R>(&mut self, range: R) -> Array<BorrowedMut<T>>
        where R: range::RangeArgument<usize>
    {
        let (start, end) = check_range(self.len, range);
        unsafe { Array::from_raw_parts_mut(self.as_mut_ptr().offset(start as isize), end - start) }
    }

    pub fn split_at_mut(&mut self, mid: usize) -> (Array<BorrowedMut<T>>, Array<BorrowedMut<T>>) {
        let mid = check_mid(self.len(), mid);
        unsafe {
            (Array::from_raw_parts_mut(self.as_mut_ptr(), mid),
             Array::from_raw_parts_mut(self.as_mut_ptr().offset(mid as isize), self.len() - mid))
        }
    }
}

impl<T> Array<Owned<T>> {
    pub fn new(len: usize) -> Result<Array<Owned<T>>> {
        Ok(Array {
               ptr: Owned::new(len)?,
               len,
           })
    }
}

impl<'a, T> Array<Borrowed<'a, T>> {
    pub unsafe fn from_raw_parts(ptr: *const T, len: usize) -> Array<Borrowed<'a, T>> {
        Array {
            ptr: Borrowed::from_raw(ptr),
            len,
        }
    }
}

impl<'a, T> Array<BorrowedMut<'a, T>> {
    pub unsafe fn from_raw_parts_mut(ptr: *mut T, len: usize) -> Array<BorrowedMut<'a, T>> {
        Array {
            ptr: BorrowedMut::from_raw_mut(ptr),
            len,
        }
    }
}

fn check_range<R>(len: usize, range: R) -> (usize, usize)
    where R: range::RangeArgument<usize>
{
    let start = match range.start() {
        Bound::Excluded(start) => start + 1,
        Bound::Included(start) => *start,
        Bound::Unbounded => 0,
    };
    let end = match range.end() {
        Bound::Excluded(end) => *end,
        Bound::Included(end) => end + 1,
        Bound::Unbounded => len,
    };
    assert!(start < end);
    assert!(end <= len);
    (start, end)
}

fn check_mid(len: usize, mid: usize) -> usize {
    assert!(0 < mid);
    assert!(mid < len);
    mid
}
