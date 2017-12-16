use super::Array;

#[test]
fn slice_range() {
    let m = Array::new(16).unwrap();
    let s = m.slice(4..12);
    assert_eq!(s.as_ptr(), unsafe { m.as_ptr().offset(4) });
    assert_eq!(s.len(), 12 - 4);
}

#[test]
fn slice_range_full() {
    let m = Array::new(16).unwrap();
    let s = m.slice(..);
    assert_eq!(s.as_ptr(), m.as_ptr());
    assert_eq!(s.len(), 16);
}

#[test]
fn slice_range_to() {
    let m = Array::new(16).unwrap();
    let s = m.slice(..12);
    assert_eq!(s.as_ptr(), m.as_ptr());
    assert_eq!(s.len(), 12);
}

#[test]
fn slice_range_from() {
    let m = Array::new(16).unwrap();
    let s = m.slice(4..);
    assert_eq!(s.as_ptr(), unsafe { m.as_ptr().offset(4) });
    assert_eq!(s.len(), 16 - 4);
}

#[test]
#[should_panic(expected = "assersion failed")]
fn slice_range_invalid_interval() {
    let m = Array::new(16).unwrap();
    m.slice(8..8);
}

#[test]
#[should_panic(expected = "assersion failed")]
fn slice_range_invalid_end() {
    let m = Array::new(16).unwrap();
    m.slice(4..17);
}

#[test]
#[should_panic(expected = "assersion failed")]
fn slice_range_to_invalid() {
    let m = Array::new(16).unwrap();
    m.slice(..17);
}

#[test]
#[should_panic(expected = "assersion failed")]
fn slice_range_from_invalid() {
    let m = Array::new(16).unwrap();
    m.slice(17..);
}

#[test]
fn slice_mut_range() {
    let mut m = Array::new(16).unwrap();
    let p = m.as_mut_ptr();
    let mut s = m.slice_mut(4..12);
    assert_eq!(s.as_ptr(), unsafe { p.offset(4) });
    assert_eq!(s.as_mut_ptr(), unsafe { p.offset(4) });
    assert_eq!(s.len(), 12 - 4);
}

#[test]
fn slice_mut_range_full() {
    let mut m = Array::new(16).unwrap();
    let p = m.as_mut_ptr();
    let mut s = m.slice_mut(..);
    assert_eq!(s.as_ptr(), p);
    assert_eq!(s.as_mut_ptr(), p);
    assert_eq!(s.len(), 16);
}

#[test]
fn slice_mut_range_to() {
    let mut m = Array::new(16).unwrap();
    let p = m.as_mut_ptr();
    let mut s = m.slice_mut(..12);
    assert_eq!(s.as_ptr(), p);
    assert_eq!(s.as_mut_ptr(), p);
    assert_eq!(s.len(), 12);
}

#[test]
fn slice_mut_range_from() {
    let mut m = Array::new(16).unwrap();
    let p = m.as_mut_ptr();
    let mut s = m.slice_mut(4..);
    assert_eq!(s.as_ptr(), unsafe { p.offset(4) });
    assert_eq!(s.as_mut_ptr(), unsafe { p.offset(4) });
    assert_eq!(s.len(), 16 - 4);
}

#[test]
#[should_panic(expected = "assersion failed")]
fn slice_mut_range_invalid_interval() {
    let mut m = Array::new(16).unwrap();
    m.slice_mut(8..8);
}

#[test]
#[should_panic(expected = "assersion failed")]
fn slice_mut_range_invalid_end() {
    let mut m = Array::new(16).unwrap();
    m.slice_mut(4..17);
}

#[test]
#[should_panic(expected = "assersion failed")]
fn slice_mut_range_to_invalid() {
    let mut m = Array::new(16).unwrap();
    m.slice_mut(..17);
}

#[test]
#[should_panic(expected = "assersion failed")]
fn slice_mut_range_from_invalid() {
    let mut m = Array::new(16).unwrap();
    m.slice_mut(17..);
}
