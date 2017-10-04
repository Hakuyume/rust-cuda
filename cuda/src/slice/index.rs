use std::mem;
use std::ops;

use super::Repr;
use super::Slice;
use super::from_raw_parts;
use super::from_raw_parts_mut;

impl<T> Slice<T> {
    fn get_slice(&self, start: Option<usize>, end: Option<usize>) -> &Slice<T> {
        unsafe {
            let repr = mem::transmute::<&Slice<T>, Repr<T>>(self);
            let start = start.unwrap_or(0);
            let end = end.unwrap_or(repr.len);
            assert!(start <= end);
            assert!(end <= repr.len);
            from_raw_parts(repr.ptr.offset(start as isize), end - start)
        }
    }

    fn get_slice_mut(&mut self, start: Option<usize>, end: Option<usize>) -> &mut Slice<T> {
        unsafe {
            let repr = mem::transmute::<&Slice<T>, Repr<T>>(self);
            let start = start.unwrap_or(0);
            let end = end.unwrap_or(repr.len);
            assert!(start <= end);
            assert!(end <= repr.len);
            from_raw_parts_mut(repr.ptr.offset(start as isize), end - start)
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
        self.get_slice_mut(None, None)
    }
}

impl<T> ops::IndexMut<ops::Range<usize>> for Slice<T> {
    fn index_mut(&mut self, index: ops::Range<usize>) -> &mut Slice<T> {
        self.get_slice_mut(Some(index.start), Some(index.end))
    }
}

impl<T> ops::IndexMut<ops::RangeFrom<usize>> for Slice<T> {
    fn index_mut(&mut self, index: ops::RangeFrom<usize>) -> &mut Slice<T> {
        self.get_slice_mut(Some(index.start), None)
    }
}

impl<T> ops::IndexMut<ops::RangeTo<usize>> for Slice<T> {
    fn index_mut(&mut self, index: ops::RangeTo<usize>) -> &mut Slice<T> {
        self.get_slice_mut(None, Some(index.end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::mem;

    #[test]
    fn index_range_full() {
        unsafe {
            let s = from_raw_parts(32 as *const f32, 16);
            let s = &s[..];
            assert_eq!(s.as_ptr(), 32 as *const f32);
            assert_eq!(s.len(), 16);
        }
    }

    #[test]
    fn index_range() {
        unsafe {
            let s = from_raw_parts(32 as *const f32, 16);
            let s = &s[4..12];
            assert_eq!(s.as_ptr(), (32 + mem::size_of::<f32>() * 4) as *const f32);
            assert_eq!(s.len(), 12 - 4);
        }
    }

    #[test]
    fn index_range_from() {
        unsafe {
            let s = from_raw_parts(32 as *const f32, 16);
            let s = &s[4..];
            assert_eq!(s.as_ptr(), (32 + mem::size_of::<f32>() * 4) as *const f32);
            assert_eq!(s.len(), 16 - 4);
        }
    }

    #[test]
    fn index_range_to() {
        unsafe {
            let s = from_raw_parts(32 as *const f32, 16);
            let s = &s[..12];
            assert_eq!(s.as_ptr(), 32 as *const f32);
            assert_eq!(s.len(), 12);
        }
    }

    #[test]
    fn index_mut_range_full() {
        unsafe {
            let s = from_raw_parts_mut(32 as *mut f32, 16);
            let s = &mut s[..];
            assert_eq!(s.as_ptr(), 32 as *const f32);
            assert_eq!(s.as_mut_ptr(), 32 as *mut f32);
            assert_eq!(s.len(), 16);
        }
    }

    #[test]
    fn index_mut_range() {
        unsafe {
            let s = from_raw_parts_mut(32 as *mut f32, 16);
            let s = &mut s[4..12];
            assert_eq!(s.as_ptr(), (32 + mem::size_of::<f32>() * 4) as *const f32);
            assert_eq!(s.as_mut_ptr(), (32 + mem::size_of::<f32>() * 4) as *mut f32);
            assert_eq!(s.len(), 12 - 4);
        }
    }

    #[test]
    fn index_mut_range_from() {
        unsafe {
            let s = from_raw_parts_mut(32 as *mut f32, 16);
            let s = &mut s[4..];
            assert_eq!(s.as_ptr(), (32 + mem::size_of::<f32>() * 4) as *const f32);
            assert_eq!(s.as_mut_ptr(), (32 + mem::size_of::<f32>() * 4) as *mut f32);
            assert_eq!(s.len(), 16 - 4);
        }
    }

    #[test]
    fn index_mut_range_to() {
        unsafe {
            let s = from_raw_parts_mut(32 as *mut f32, 16);
            let s = &mut s[..12];
            assert_eq!(s.as_ptr(), 32 as *const f32);
            assert_eq!(s.as_mut_ptr(), 32 as *mut f32);
            assert_eq!(s.len(), 12);
        }
    }
}
