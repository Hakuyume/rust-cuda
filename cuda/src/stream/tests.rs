use super::Stream;

#[test]
fn new() {
    let stream = Stream::new().unwrap();
    assert!(!stream.as_ptr().is_null());
}

#[test]
fn synchronize() {
    let stream = Stream::new().unwrap();
    stream.synchronize().unwrap();
}

#[test]
fn default() {
    let stream = Stream::default();
    assert!(stream.as_ptr().is_null());
}
