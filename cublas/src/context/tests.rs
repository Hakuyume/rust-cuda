use super::Context;
use super::PointerMode;

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

#[test]
fn set_pointer_mode() {
    let mut context = Context::new().unwrap();

    context.set_pointer_mode(PointerMode::Host).unwrap();
    assert_eq!(context.get_pointer_mode().unwrap(), PointerMode::Host);

    context.set_pointer_mode(PointerMode::Device).unwrap();
    assert_eq!(context.get_pointer_mode().unwrap(), PointerMode::Device);
}
