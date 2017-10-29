use super::{View, ViewMut};

#[test]
fn from_raw_parts() {
    unsafe {
        let s = super::from_raw_parts(32 as *const f32, 16);
        assert_eq!(s.as_ptr(), 32 as *const f32);
        assert_eq!(s.len(), 16);
    }
}

#[test]
fn from_raw_parts_mut() {
    unsafe {
        let mut s = super::from_raw_parts_mut(32 as *mut f32, 16);
        assert_eq!(s.as_ptr(), 32 as *const f32);
        assert_eq!(s.as_mut_ptr(), 32 as *mut f32);
        assert_eq!(s.len(), 16);
    }
}
