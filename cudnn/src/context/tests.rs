use super::Context;

#[test]
fn new() {
    let mut context = Context::new().unwrap();
    assert!(!context.as_mut_ptr().is_null());
}
