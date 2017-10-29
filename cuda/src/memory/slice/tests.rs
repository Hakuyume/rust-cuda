use std::mem;

use super::{Repr, ReprMut};
use super::{Slice, SliceMut};

#[test]
fn slice_range_full() {
    unsafe {
        let s = super::from_raw_parts(32 as *const f32, 16);
        let s = s.slice(..);
        assert_eq!(s.as_ptr(), 32 as *const f32);
        assert_eq!(s.len(), 16);
    }
}

#[test]
fn slice_range() {
    unsafe {
        let s = super::from_raw_parts(32 as *const f32, 16);
        let s = s.slice(4..12);
        assert_eq!(s.as_ptr(), (32 + mem::size_of::<f32>() * 4) as *const f32);
        assert_eq!(s.len(), 12 - 4);
    }
}

#[test]
fn slice_range_from() {
    unsafe {
        let s = super::from_raw_parts(32 as *const f32, 16);
        let s = s.slice(4..);
        assert_eq!(s.as_ptr(), (32 + mem::size_of::<f32>() * 4) as *const f32);
        assert_eq!(s.len(), 16 - 4);
    }
}

#[test]
fn slice_range_to() {
    unsafe {
        let s = super::from_raw_parts(32 as *const f32, 16);
        let s = s.slice(..12);
        assert_eq!(s.as_ptr(), 32 as *const f32);
        assert_eq!(s.len(), 12);
    }
}

#[test]
#[should_panic]
fn slice_range_invalid() {
    unsafe {
        let s = super::from_raw_parts(32 as *const f32, 16);
        s.slice(8..24);
    }
}

#[test]
#[should_panic]
fn slice_range_from_invalid() {
    unsafe {
        let s = super::from_raw_parts(32 as *const f32, 16);
        s.slice(24..);
    }
}

#[test]
#[should_panic]
fn slice_range_to_invalid() {
    unsafe {
        let s = super::from_raw_parts(32 as *const f32, 16);
        s.slice(..24);
    }
}

#[test]
fn slice_mut_range_full() {
    unsafe {
        let mut s = super::from_raw_parts_mut(32 as *mut f32, 16);
        let mut s = s.slice_mut(..);
        assert_eq!(s.as_ptr(), 32 as *const f32);
        assert_eq!(s.as_mut_ptr(), 32 as *mut f32);
        assert_eq!(s.len(), 16);
    }
}

#[test]
fn slice_mut_range() {
    unsafe {
        let mut s = super::from_raw_parts_mut(32 as *mut f32, 16);
        let mut s = s.slice_mut(..12);
        assert_eq!(s.as_ptr(), (32 + mem::size_of::<f32>() * 4) as *const f32);
        assert_eq!(s.as_mut_ptr(), (32 + mem::size_of::<f32>() * 4) as *mut f32);
        assert_eq!(s.len(), 12 - 4);
    }
}

#[test]
fn slice_mut_range_from() {
    unsafe {
        let mut s = super::from_raw_parts_mut(32 as *mut f32, 16);
        let mut s = s.slice_mut(4..);
        assert_eq!(s.as_ptr(), (32 + mem::size_of::<f32>() * 4) as *const f32);
        assert_eq!(s.as_mut_ptr(), (32 + mem::size_of::<f32>() * 4) as *mut f32);
        assert_eq!(s.len(), 16 - 4);
    }
}

#[test]
fn slice_mut_range_to() {
    unsafe {
        let mut s = super::from_raw_parts_mut(32 as *mut f32, 16);
        let mut s = s.slice_mut(..12);
        assert_eq!(s.as_ptr(), 32 as *const f32);
        assert_eq!(s.as_mut_ptr(), 32 as *mut f32);
        assert_eq!(s.len(), 12);
    }
}

#[test]
#[should_panic]
fn slice_mut_range_invalid() {
    unsafe {
        let mut s = super::from_raw_parts_mut(32 as *mut f32, 16);
        s.slice_mut(8..24);
    }
}

#[test]
#[should_panic]
fn slice_mut_range_from_invalid() {
    unsafe {
        let mut s = super::from_raw_parts_mut(32 as *mut f32, 16);
        s.slice_mut(24..);
    }
}

#[test]
#[should_panic]
fn slice_mut_range_to_invalid() {
    unsafe {
        let mut s = super::from_raw_parts_mut(32 as *mut f32, 16);
        s.slice_mut(..24);
    }
}
