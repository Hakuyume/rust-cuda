use super::{Slice, SliceMut};
use super::{from_raw_parts, from_raw_parts_mut};

pub trait View<T> {
    fn as_ptr(&self) -> *const T;
    fn len(&self) -> usize;

    fn slice<'a>(&'a self, start: usize, end: usize) -> Slice<'a, T> {
        assert!(start <= end);
        assert!(end <= self.len());
        unsafe { from_raw_parts(self.as_ptr().offset(start as isize), end - start) }
    }
    fn split_at<'a>(&'a self, mid: usize) -> (Slice<'a, T>, Slice<'a, T>) {
        assert!(mid <= self.len());
        unsafe {
            (from_raw_parts(self.as_ptr(), mid),
             from_raw_parts(self.as_ptr().offset(mid as isize), self.len() - mid))
        }
    }
}

pub trait ViewMut<T>: View<T> {
    fn as_mut_ptr(&mut self) -> *mut T;

    fn slice_mut<'a>(&'a mut self, start: usize, end: usize) -> SliceMut<'a, T> {
        assert!(start <= end);
        assert!(end <= self.len());
        unsafe { from_raw_parts_mut(self.as_mut_ptr().offset(start as isize), end - start) }
    }
    fn split_at_mut<'a>(&'a mut self, mid: usize) -> (SliceMut<'a, T>, SliceMut<'a, T>) {
        assert!(mid <= self.len());
        unsafe {
            (from_raw_parts_mut(self.as_mut_ptr(), mid),
             from_raw_parts_mut(self.as_mut_ptr().offset(mid as isize), self.len() - mid))
        }
    }
}
