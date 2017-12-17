use super::Stream;

#[test]
fn new() {
    Stream::new().unwrap();
}

#[test]
fn with() {
    let stream = Stream::new().unwrap();
    stream.with(|s| assert!(!s.as_ptr().is_null()));
}

#[test]
fn synchronize() {
    let stream = Stream::new().unwrap();
    let sync_handle = stream.with(|s| assert!(!s.as_ptr().is_null()));
    sync_handle.synchronize().unwrap();
}
