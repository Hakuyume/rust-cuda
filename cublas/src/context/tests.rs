use super::Context;

#[test]
fn new() {
    let mut context = Context::new().unwrap();
    assert!(!context.as_mut_ptr().is_null());
}

#[test]
fn get_pointer_mode() {
    let context = Context::new().unwrap();
    context.get_pointer_mode().unwrap();
}
