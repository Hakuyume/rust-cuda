use std::mem;
use std::ops;
use super::{Repr, Slice};

impl<T> Slice<T> {
    fn get_slice(&self, start: Option<usize>, end: Option<usize>) -> &Slice<T> {
        unsafe {
            let repr = mem::transmute::<&Slice<T>, Repr<T>>(self);
            let start = start.unwrap_or(0);
            let end = end.unwrap_or(repr.len);
            assert!(start <= end);
            assert!(end <= repr.len);
            Slice::new(repr.ptr.offset(start as isize), end - start)
        }
    }

    fn get_mut_slice(&mut self, start: Option<usize>, end: Option<usize>) -> &mut Slice<T> {
        unsafe {
            let repr = mem::transmute::<&Slice<T>, Repr<T>>(self);
            let start = start.unwrap_or(0);
            let end = end.unwrap_or(repr.len);
            assert!(start <= end);
            assert!(end <= repr.len);
            Slice::new_mut(repr.ptr.offset(start as isize), end - start)
        }
    }
}

impl<T> ops::Index<ops::RangeFull> for Slice<T> {
    type Output = Slice<T>;
    fn index(&self, _: ops::RangeFull) -> &Slice<T> {
        self.get_slice(None, None)
    }
}

impl<T> ops::Index<ops::Range<usize>> for Slice<T> {
    type Output = Slice<T>;
    fn index(&self, index: ops::Range<usize>) -> &Slice<T> {
        self.get_slice(Some(index.start), Some(index.end))
    }
}

impl<T> ops::Index<ops::RangeFrom<usize>> for Slice<T> {
    type Output = Slice<T>;
    fn index(&self, index: ops::RangeFrom<usize>) -> &Slice<T> {
        self.get_slice(Some(index.start), None)
    }
}

impl<T> ops::Index<ops::RangeTo<usize>> for Slice<T> {
    type Output = Slice<T>;
    fn index(&self, index: ops::RangeTo<usize>) -> &Slice<T> {
        self.get_slice(None, Some(index.end))
    }
}

impl<T> ops::IndexMut<ops::RangeFull> for Slice<T> {
    fn index_mut(&mut self, _: ops::RangeFull) -> &mut Slice<T> {
        self.get_mut_slice(None, None)
    }
}

impl<T> ops::IndexMut<ops::Range<usize>> for Slice<T> {
    fn index_mut(&mut self, index: ops::Range<usize>) -> &mut Slice<T> {
        self.get_mut_slice(Some(index.start), Some(index.end))
    }
}

impl<T> ops::IndexMut<ops::RangeFrom<usize>> for Slice<T> {
    fn index_mut(&mut self, index: ops::RangeFrom<usize>) -> &mut Slice<T> {
        self.get_mut_slice(Some(index.start), None)
    }
}

impl<T> ops::IndexMut<ops::RangeTo<usize>> for Slice<T> {
    fn index_mut(&mut self, index: ops::RangeTo<usize>) -> &mut Slice<T> {
        self.get_mut_slice(None, Some(index.end))
    }
}
