use std::mem;

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
        let s = super::from_raw_parts_mut(32 as *mut f32, 16);
        assert_eq!(s.as_ptr(), 32 as *const f32);
        assert_eq!(s.as_mut_ptr(), 32 as *mut f32);
        assert_eq!(s.len(), 16);
    }
}

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
        let s = super::from_raw_parts_mut(32 as *mut f32, 16);
        let (l, r) = s.split_at_mut(4);
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
        let s = super::from_raw_parts_mut(32 as *mut f32, 16);
        s.split_at_mut(24);
    }
}

#[test]
fn index_range_full() {
    unsafe {
        let s = super::from_raw_parts(32 as *const f32, 16);
        let s = &s[..];
        assert_eq!(s.as_ptr(), 32 as *const f32);
        assert_eq!(s.len(), 16);
    }
}

#[test]
fn index_range() {
    unsafe {
        let s = super::from_raw_parts(32 as *const f32, 16);
        let s = &s[4..12];
        assert_eq!(s.as_ptr(), (32 + mem::size_of::<f32>() * 4) as *const f32);
        assert_eq!(s.len(), 12 - 4);
    }
}

#[test]
fn index_range_from() {
    unsafe {
        let s = super::from_raw_parts(32 as *const f32, 16);
        let s = &s[4..];
        assert_eq!(s.as_ptr(), (32 + mem::size_of::<f32>() * 4) as *const f32);
        assert_eq!(s.len(), 16 - 4);
    }
}

#[test]
fn index_range_to() {
    unsafe {
        let s = super::from_raw_parts(32 as *const f32, 16);
        let s = &s[..12];
        assert_eq!(s.as_ptr(), 32 as *const f32);
        assert_eq!(s.len(), 12);
    }
}

#[test]
#[should_panic]
fn index_range_invalid() {
    unsafe {
        let s = super::from_raw_parts(32 as *const f32, 16);
        &s[8..24];
    }
}

#[test]
#[should_panic]
fn index_range_from_invalid() {
    unsafe {
        let s = super::from_raw_parts(32 as *const f32, 16);
        &s[24..];
    }
}

#[test]
#[should_panic]
fn index_range_to_invalid() {
    unsafe {
        let s = super::from_raw_parts(32 as *const f32, 16);
        &s[..24];
    }
}

#[test]
fn index_mut_range_full() {
    unsafe {
        let s = super::from_raw_parts_mut(32 as *mut f32, 16);
        let s = &mut s[..];
        assert_eq!(s.as_ptr(), 32 as *const f32);
        assert_eq!(s.as_mut_ptr(), 32 as *mut f32);
        assert_eq!(s.len(), 16);
    }
}

#[test]
fn index_mut_range() {
    unsafe {
        let s = super::from_raw_parts_mut(32 as *mut f32, 16);
        let s = &mut s[4..12];
        assert_eq!(s.as_ptr(), (32 + mem::size_of::<f32>() * 4) as *const f32);
        assert_eq!(s.as_mut_ptr(), (32 + mem::size_of::<f32>() * 4) as *mut f32);
        assert_eq!(s.len(), 12 - 4);
    }
}

#[test]
fn index_mut_range_from() {
    unsafe {
        let s = super::from_raw_parts_mut(32 as *mut f32, 16);
        let s = &mut s[4..];
        assert_eq!(s.as_ptr(), (32 + mem::size_of::<f32>() * 4) as *const f32);
        assert_eq!(s.as_mut_ptr(), (32 + mem::size_of::<f32>() * 4) as *mut f32);
        assert_eq!(s.len(), 16 - 4);
    }
}

#[test]
fn index_mut_range_to() {
    unsafe {
        let s = super::from_raw_parts_mut(32 as *mut f32, 16);
        let s = &mut s[..12];
        assert_eq!(s.as_ptr(), 32 as *const f32);
        assert_eq!(s.as_mut_ptr(), 32 as *mut f32);
        assert_eq!(s.len(), 12);
    }
}

#[test]
#[should_panic]
fn index_mut_range_invalid() {
    unsafe {
        let s = super::from_raw_parts_mut(32 as *mut f32, 16);
        &mut s[8..24];
    }
}

#[test]
#[should_panic]
fn index_mut_range_from_invalid() {
    unsafe {
        let s = super::from_raw_parts_mut(32 as *mut f32, 16);
        &mut s[24..];
    }
}

#[test]
#[should_panic]
fn index_mut_range_to_invalid() {
    unsafe {
        let s = super::from_raw_parts_mut(32 as *mut f32, 16);
        &mut s[..24];
    }
}
