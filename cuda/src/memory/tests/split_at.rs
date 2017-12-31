use super::Array;

#[test]
fn split_at() {
    let m = Array::<u8>::new(16).unwrap();
    let (l, r) = m.split_at(4);
    assert_eq!(l.as_ptr(), m.as_ptr());
    assert_eq!(l.len(), 4);
    assert_eq!(r.as_ptr(), unsafe { m.as_ptr().offset(4) });
    assert_eq!(r.len(), 16 - 4);
}

#[test]
#[should_panic(expected = "assertion failed")]
fn split_at_invalid() {
    let m = Array::<u8>::new(16).unwrap();
    m.split_at(24);
}

#[test]
fn split_at_mut() {
    let mut m = Array::<u8>::new(16).unwrap();
    let p = m.as_mut_ptr();
    let (mut l, mut r) = m.split_at_mut(4);
    assert_eq!(l.as_ptr(), p);
    assert_eq!(l.as_mut_ptr(), p);
    assert_eq!(l.len(), 4);
    assert_eq!(r.as_ptr(), unsafe { p.offset(4) });
    assert_eq!(r.as_mut_ptr(), unsafe { p.offset(4) });
    assert_eq!(r.len(), 16 - 4);
}

#[test]
#[should_panic(expected = "assertion failed")]
fn split_at_mut_invalid() {
    let mut m = Array::<u8>::new(16).unwrap();
    m.split_at_mut(24);
}
