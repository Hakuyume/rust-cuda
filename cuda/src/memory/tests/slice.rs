use super::Array;

#[test]
fn slice_range() {
    let arr = Array::new(16).unwrap();
    let s = arr.slice(4..12);
    assert_eq!(s.as_ptr(), unsafe { arr.as_ptr().offset(4) });
    assert_eq!(s.len(), 12 - 4);
}

#[test]
fn slice_range_full() {
    let arr = Array::new(16).unwrap();
    let s = arr.slice(..);
    assert_eq!(s.as_ptr(), arr.as_ptr());
    assert_eq!(s.len(), 16);
}

#[test]
fn slice_range_to() {
    let arr = Array::new(16).unwrap();
    let s = arr.slice(..12);
    assert_eq!(s.as_ptr(), arr.as_ptr());
    assert_eq!(s.len(), 12);
}

#[test]
fn slice_range_from() {
    let arr = Array::new(16).unwrap();
    let s = arr.slice(4..);
    assert_eq!(s.as_ptr(), unsafe { arr.as_ptr().offset(4) });
    assert_eq!(s.len(), 16 - 4);
}

#[test]
#[should_panic(expected = "assertion failed")]
fn slice_range_invalid_interval() {
    let arr = Array::new(16).unwrap();
    arr.slice(8..8);
}

#[test]
#[should_panic(expected = "assertion failed")]
fn slice_range_invalid_end() {
    let arr = Array::new(16).unwrap();
    arr.slice(4..17);
}

#[test]
#[should_panic(expected = "assertion failed")]
fn slice_range_to_invalid() {
    let arr = Array::new(16).unwrap();
    arr.slice(..17);
}

#[test]
#[should_panic(expected = "assertion failed")]
fn slice_range_from_invalid() {
    let arr = Array::new(16).unwrap();
    arr.slice(17..);
}

#[test]
fn slice_mut_range() {
    let mut arr = Array::new(16).unwrap();
    let p = arr.as_mut_ptr();
    let mut s = arr.slice_mut(4..12);
    assert_eq!(s.as_ptr(), unsafe { p.offset(4) });
    assert_eq!(s.as_mut_ptr(), unsafe { p.offset(4) });
    assert_eq!(s.len(), 12 - 4);
}

#[test]
fn slice_mut_range_full() {
    let mut arr = Array::new(16).unwrap();
    let p = arr.as_mut_ptr();
    let mut s = arr.slice_mut(..);
    assert_eq!(s.as_ptr(), p);
    assert_eq!(s.as_mut_ptr(), p);
    assert_eq!(s.len(), 16);
}

#[test]
fn slice_mut_range_to() {
    let mut arr = Array::new(16).unwrap();
    let p = arr.as_mut_ptr();
    let mut s = arr.slice_mut(..12);
    assert_eq!(s.as_ptr(), p);
    assert_eq!(s.as_mut_ptr(), p);
    assert_eq!(s.len(), 12);
}

#[test]
fn slice_mut_range_from() {
    let mut arr = Array::new(16).unwrap();
    let p = arr.as_mut_ptr();
    let mut s = arr.slice_mut(4..);
    assert_eq!(s.as_ptr(), unsafe { p.offset(4) });
    assert_eq!(s.as_mut_ptr(), unsafe { p.offset(4) });
    assert_eq!(s.len(), 16 - 4);
}

#[test]
#[should_panic(expected = "assertion failed")]
fn slice_mut_range_invalid_interval() {
    let mut arr = Array::new(16).unwrap();
    arr.slice_mut(8..8);
}

#[test]
#[should_panic(expected = "assertion failed")]
fn slice_mut_range_invalid_end() {
    let mut arr = Array::new(16).unwrap();
    arr.slice_mut(4..17);
}

#[test]
#[should_panic(expected = "assertion failed")]
fn slice_mut_range_to_invalid() {
    let mut arr = Array::new(16).unwrap();
    arr.slice_mut(..17);
}

#[test]
#[should_panic(expected = "assertion failed")]
fn slice_mut_range_from_invalid() {
    let mut arr = Array::new(16).unwrap();
    arr.slice_mut(17..);
}
