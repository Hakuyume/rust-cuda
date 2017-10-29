use super::{BorrowedView, BorrowedViewMut};
use super::{from_raw_parts, from_raw_parts_mut};

pub trait View<T> {
    fn as_ptr(&self) -> *const T;
    fn len(&self) -> usize;

    fn split_at<'a>(&'a self, mid: usize) -> (BorrowedView<'a, T>, BorrowedView<'a, T>) {
        assert!(mid <= self.len());
        unsafe {
            (from_raw_parts(self.as_ptr(), mid),
             from_raw_parts(self.as_ptr().offset(mid as isize), self.len() - mid))
        }
    }
}

pub trait ViewMut<T>: View<T> {
    fn as_mut_ptr(&mut self) -> *mut T;

    fn split_at_mut<'a>(&'a mut self,
                        mid: usize)
                        -> (BorrowedViewMut<'a, T>, BorrowedViewMut<'a, T>) {
        assert!(mid <= self.len());
        unsafe {
            (from_raw_parts_mut(self.as_mut_ptr(), mid),
             from_raw_parts_mut(self.as_mut_ptr().offset(mid as isize), self.len() - mid))
        }
    }
}

#[cfg(test)]
mod tests;
