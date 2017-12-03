use tensor::Format;

use super::Descriptor;

#[test]
fn new() {
    Descriptor::<f32>::new().unwrap();
}

#[test]
fn set_4d_nchw() {
    let mut desc = Descriptor::<f32>::new().unwrap();
    desc.set_4d(Format::NCHW, 2, 3, 5, 7).unwrap();
}

#[test]
fn set_4d_nhwc() {
    let mut desc = Descriptor::<f32>::new().unwrap();
    desc.set_4d(Format::NHWC, 2, 3, 5, 7).unwrap();
}
