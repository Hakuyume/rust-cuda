use std::ops;

use super::{Repr, ReprMut};
use super::{View, ViewMut};
use super::{from_raw_parts, from_raw_parts_mut};

pub trait Slice<T, S> {
    fn slice<'a>(&'a self, slice: S) -> View<'a, T>;
}

fn get_slice<T, R>(repr: &R, start: Option<usize>, end: Option<usize>) -> View<T>
    where R: Repr<T>
{
    let start = start.unwrap_or(0);
    let end = end.unwrap_or(repr.len());
    assert!(start <= end);
    assert!(end <= repr.len());
    unsafe { from_raw_parts(repr.as_ptr().offset(start as isize), end - start) }
}

impl<T, R> Slice<T, ops::RangeFull> for R
    where R: Repr<T>
{
    fn slice<'a>(&'a self, _: ops::RangeFull) -> View<'a, T> {
        get_slice(self, None, None)
    }
}

impl<T, R> Slice<T, ops::Range<usize>> for R
    where R: Repr<T>
{
    fn slice<'a>(&'a self, slice: ops::Range<usize>) -> View<'a, T> {
        get_slice(self, Some(slice.start), Some(slice.end))
    }
}

impl<T, R> Slice<T, ops::RangeFrom<usize>> for R
    where R: Repr<T>
{
    fn slice<'a>(&'a self, slice: ops::RangeFrom<usize>) -> View<'a, T> {
        get_slice(self, Some(slice.start), None)
    }
}

impl<T, R> Slice<T, ops::RangeTo<usize>> for R
    where R: Repr<T>
{
    fn slice<'a>(&'a self, slice: ops::RangeTo<usize>) -> View<'a, T> {
        get_slice(self, None, Some(slice.end))
    }
}

pub trait SliceMut<T, S>: Slice<T, S> {
    fn slice_mut<'a>(&'a mut self, slice: S) -> ViewMut<'a, T>;
}

fn get_slice_mut<T, R>(repr: &mut R, start: Option<usize>, end: Option<usize>) -> ViewMut<T>
    where R: ReprMut<T>
{
    let start = start.unwrap_or(0);
    let end = end.unwrap_or(repr.len());
    assert!(start <= end);
    assert!(end <= repr.len());
    unsafe { from_raw_parts_mut(repr.as_mut_ptr().offset(start as isize), end - start) }
}

impl<T, R> SliceMut<T, ops::RangeFull> for R
    where R: ReprMut<T>
{
    fn slice_mut<'a>(&'a mut self, _: ops::RangeFull) -> ViewMut<'a, T> {
        get_slice_mut(self, None, None)
    }
}

impl<T, R> SliceMut<T, ops::Range<usize>> for R
    where R: ReprMut<T>
{
    fn slice_mut<'a>(&'a mut self, slice: ops::Range<usize>) -> ViewMut<'a, T> {
        get_slice_mut(self, Some(slice.start), Some(slice.end))
    }
}

impl<T, R> SliceMut<T, ops::RangeFrom<usize>> for R
    where R: ReprMut<T>
{
    fn slice_mut<'a>(&'a mut self, slice: ops::RangeFrom<usize>) -> ViewMut<'a, T> {
        get_slice_mut(self, Some(slice.start), None)
    }
}

impl<T, R> SliceMut<T, ops::RangeTo<usize>> for R
    where R: ReprMut<T>
{
    fn slice_mut<'a>(&'a mut self, slice: ops::RangeTo<usize>) -> ViewMut<'a, T> {
        get_slice_mut(self, None, Some(slice.end))
    }
}

#[cfg(test)]
mod tests;
