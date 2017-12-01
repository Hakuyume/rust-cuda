use std::mem;

use super::Descriptor;
use super::Format;

#[test]
fn set_4d_nchw() {
    let desc: Descriptor<f32> = Descriptor::new().unwrap();
    desc.set_4d(Format::NCHW, 2, 3, 5, 7).unwrap();
    assert_eq!(desc.len(), 2 * 3 * 5 * 7);
    assert_eq!(desc.get_size().unwrap(),
               mem::size_of::<f32>() * 2 * 3 * 5 * 7);
    assert_eq!(desc.get_4d().unwrap(), (2, 3, 5, 7, 3 * 5 * 7, 5 * 7, 7, 1));
}

#[test]
fn set_4d_nhwc() {
    let desc: Descriptor<f32> = Descriptor::new().unwrap();
    desc.set_4d(Format::NHWC, 2, 3, 5, 7).unwrap();
    assert_eq!(desc.len(), 2 * 3 * 5 * 7);
    assert_eq!(desc.get_size().unwrap(),
               mem::size_of::<f32>() * 2 * 3 * 5 * 7);
    assert_eq!(desc.get_4d().unwrap(), (2, 7, 3, 5, 3 * 5 * 7, 1, 5 * 7, 7));
}

#[test]
fn set_4d_ex() {
    let desc: Descriptor<f32> = Descriptor::new().unwrap();
    desc.set_4d_ex(2, 3, 5, 7, 3 * 5 * 7 * 16, 5 * 7 * 8, 7 * 4, 2)
        .unwrap();
    assert_eq!(desc.len(), 2 * 3 * 5 * 7 * 16);
    assert_eq!(desc.get_size().unwrap(),
               mem::size_of::<f32>() * 2 * 3 * 5 * 7 * 16);
    assert_eq!(desc.get_4d().unwrap(),
               (2, 3, 5, 7, 3 * 5 * 7 * 16, 5 * 7 * 8, 7 * 4, 2));
}
