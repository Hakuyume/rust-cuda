use std::mem;

use super::{Repr, ReprMut};

#[test]
fn split_at() {
    unsafe {
        let s = super::from_raw_parts(32 as *const f32, 16);
        let (l, r) = s.split_at(4);
        assert_eq!(l.as_ptr(), 32 as *const f32);
        assert_eq!(l.len(), 4);
        assert_eq!(r.as_ptr(), (32 + mem::size_of::<f32>() * 4) as *const f32);
        assert_eq!(r.len(), 16 - 4);
    }
}

#[test]
#[should_panic]
fn split_at_invalid() {
    unsafe {
        let s = super::from_raw_parts(32 as *const f32, 16);
        s.split_at(24);
    }
}

#[test]
fn split_at_mut() {
    unsafe {
        let mut s = super::from_raw_parts_mut(32 as *mut f32, 16);
        let (mut l, mut r) = s.split_at_mut(4);
        assert_eq!(l.as_ptr(), 32 as *const f32);
        assert_eq!(l.as_mut_ptr(), 32 as *mut f32);
        assert_eq!(l.len(), 4);
        assert_eq!(r.as_ptr(), (32 + mem::size_of::<f32>() * 4) as *const f32);
        assert_eq!(r.as_mut_ptr(), (32 + mem::size_of::<f32>() * 4) as *mut f32);
        assert_eq!(r.len(), 16 - 4);
    }
}

#[test]
#[should_panic]
fn split_at_mut_invalid() {
    unsafe {
        let mut s = super::from_raw_parts_mut(32 as *mut f32, 16);
        s.split_at_mut(24);
    }
}
