use std::mem;

use cuda::memory;

use super::Descriptor;
use super::Format;
use super::Tensor;
use super::TensorMut;

#[test]
fn set_4d_nchw() {
    let mut desc: Descriptor<f32> = Descriptor::new().unwrap();
    desc.set_4d(Format::NCHW, 2, 3, 5, 7).unwrap();
    assert_eq!(desc.len(), 2 * 3 * 5 * 7);
    assert_eq!(desc.get_size().unwrap(),
               mem::size_of::<f32>() * 2 * 3 * 5 * 7);
    assert_eq!(desc.get_4d().unwrap(), (2, 3, 5, 7, 3 * 5 * 7, 5 * 7, 7, 1));
}

#[test]
fn set_4d_nhwc() {
    let mut desc: Descriptor<f32> = Descriptor::new().unwrap();
    desc.set_4d(Format::NHWC, 2, 3, 5, 7).unwrap();
    assert_eq!(desc.len(), 2 * 3 * 5 * 7);
    assert_eq!(desc.get_size().unwrap(),
               mem::size_of::<f32>() * 2 * 3 * 5 * 7);
    assert_eq!(desc.get_4d().unwrap(), (2, 3, 5, 7, 5 * 7 * 3, 1, 7 * 3, 3));
}

#[test]
fn set_4d_ex() {
    let mut desc: Descriptor<f32> = Descriptor::new().unwrap();
    desc.set_4d_ex(2, 3, 5, 7, 3 * 5 * 7 * 16, 5 * 7 * 8, 7 * 4, 2)
        .unwrap();
    assert_eq!(desc.len(), 2 * 3 * 5 * 7 * 16);
    assert_eq!(desc.get_size().unwrap(),
               mem::size_of::<f32>() * 2 * 3 * 5 * 7 * 16);
    assert_eq!(desc.get_4d().unwrap(),
               (2, 3, 5, 7, 3 * 5 * 7 * 16, 5 * 7 * 8, 7 * 4, 2));
}

#[test]
fn tensor() {
    let mut desc: Descriptor<f32> = Descriptor::new().unwrap();
    desc.set_4d(Format::NCHW, 2, 3, 5, 7).unwrap();
    let mem = memory::Memory::new(desc.len()).unwrap();
    let tensor = Tensor::new(&desc, &mem);
    assert_eq!(tensor.desc().as_ptr(), desc.as_ptr());
    assert_eq!(tensor.mem.as_ptr(), mem.as_ptr());
}

#[test]
#[should_panic]
fn tensor_invalid() {
    let mut desc: Descriptor<f32> = Descriptor::new().unwrap();
    desc.set_4d(Format::NCHW, 2, 3, 5, 7).unwrap();
    let mem = memory::Memory::new(desc.len() - 1).unwrap();
    Tensor::new(&desc, &mem);
}

#[test]
fn tensor_mut() {
    let mut desc: Descriptor<f32> = Descriptor::new().unwrap();
    desc.set_4d(Format::NCHW, 2, 3, 5, 7).unwrap();
    let mut mem = memory::Memory::new(desc.len()).unwrap();
    let mem_ptr = mem.as_ptr();
    let mem_mut_ptr = mem.as_mut_ptr();
    let mut tensor = TensorMut::new(&desc, &mut mem);
    assert_eq!(tensor.desc().as_ptr(), desc.as_ptr());
    assert_eq!(tensor.mem.as_ptr(), mem_ptr);
    assert_eq!(tensor.mem_mut.as_mut_ptr(), mem_mut_ptr);
}

#[test]
#[should_panic]
fn tensor_mut_invalid() {
    let mut desc: Descriptor<f32> = Descriptor::new().unwrap();
    desc.set_4d(Format::NCHW, 2, 3, 5, 7).unwrap();
    let mut mem = memory::Memory::new(desc.len() - 1).unwrap();
    TensorMut::new(&desc, &mut mem);
}
