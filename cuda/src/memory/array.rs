use std::collections::Bound;

use Result;
use nightly::collections::range;

use super::{Repr, ReprMut};
use super::owned_repr::OwnedRepr;
use super::borrowed_repr::{BorrowedRepr, BorrowedReprMut};

pub struct ArrayBase<R>
    where R: Repr
{
    repr: R,
    len: usize,
}

pub type Array<T> = ArrayBase<OwnedRepr<T>>;
impl<T> Array<T> {
    pub fn new(len: usize) -> Result<Array<T>> {
        Ok(ArrayBase {
               repr: OwnedRepr::new(len)?,
               len,
           })
    }
}

pub type Slice<'a, T> = ArrayBase<BorrowedRepr<'a, T>>;
impl<'a, T> Slice<'a, T> {
    pub unsafe fn from_raw_parts(ptr: *const T, len: usize) -> Slice<'a, T> {
        ArrayBase {
            repr: BorrowedRepr::from_raw(ptr),
            len,
        }
    }
}

pub type SliceMut<'a, T> = ArrayBase<BorrowedReprMut<'a, T>>;
impl<'a, T> SliceMut<'a, T> {
    pub unsafe fn from_raw_parts_mut(ptr: *mut T, len: usize) -> SliceMut<'a, T> {
        ArrayBase {
            repr: BorrowedReprMut::from_raw_mut(ptr),
            len,
        }
    }
}

impl<T, R> ArrayBase<R>
    where R: Repr<Type = T>
{
    pub fn as_ptr(&self) -> *const T {
        self.repr.as_ptr()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn slice<S>(&self, range: S) -> Slice<T>
        where S: range::RangeArgument<usize>
    {
        let (start, end) = check_range(self.len, range);
        unsafe { Slice::from_raw_parts(self.as_ptr().offset(start as isize), end - start) }
    }

    pub fn split_at(&self, mid: usize) -> (Slice<T>, Slice<T>) {
        let mid = check_mid(self.len(), mid);
        unsafe {
            (Slice::from_raw_parts(self.as_ptr(), mid),
             Slice::from_raw_parts(self.as_ptr().offset(mid as isize), self.len() - mid))
        }
    }
}

impl<T, R> ArrayBase<R>
    where R: ReprMut<Type = T>
{
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.repr.as_mut_ptr()
    }

    pub fn slice_mut<S>(&mut self, range: S) -> SliceMut<T>
        where S: range::RangeArgument<usize>
    {
        let (start, end) = check_range(self.len, range);
        unsafe {
            SliceMut::from_raw_parts_mut(self.as_mut_ptr().offset(start as isize), end - start)
        }
    }

    pub fn split_at_mut(&mut self, mid: usize) -> (SliceMut<T>, SliceMut<T>) {
        let mid = check_mid(self.len(), mid);
        unsafe {
            (SliceMut::from_raw_parts_mut(self.as_mut_ptr(), mid),
             SliceMut::from_raw_parts_mut(self.as_mut_ptr().offset(mid as isize), self.len() - mid))
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
