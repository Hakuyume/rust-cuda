use super::Stream;

#[test]
fn new() {
    Stream::new().unwrap();
}

#[test]
fn synchronize() {
    let s = Stream::new().unwrap();
    s.synchronize().unwrap();
}

#[test]
fn with() {
    let s = Stream::new().unwrap();
    s.with(|s| assert!(!s.as_ptr().is_null()));
}
