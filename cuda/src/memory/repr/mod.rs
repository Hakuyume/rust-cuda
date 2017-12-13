use super::{View, ViewMut};
use super::{from_raw_parts, from_raw_parts_mut};

pub trait Repr<T> {
    fn as_ptr(&self) -> *const T;
    fn len(&self) -> usize;

    fn split_at<'a>(&'a self, mid: usize) -> (View<'a, T>, View<'a, T>) {
        assert!(mid <= self.len());
        unsafe {
            (from_raw_parts(self.as_ptr(), mid),
             from_raw_parts(self.as_ptr().offset(mid as isize), self.len() - mid))
        }
    }
}

pub trait ReprMut<T>: Repr<T> {
    fn as_mut_ptr(&mut self) -> *mut T;

    fn split_at_mut<'a>(&'a mut self, mid: usize) -> (ViewMut<'a, T>, ViewMut<'a, T>) {
        assert!(mid <= self.len());
        unsafe {
            (from_raw_parts_mut(self.as_mut_ptr(), mid),
             from_raw_parts_mut(self.as_mut_ptr().offset(mid as isize), self.len() - mid))
        }
    }
}

#[cfg(test)]
mod tests;
