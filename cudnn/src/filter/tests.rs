use cuda::memory;
use cuda::memory::{Repr, ReprMut};

use tensor::Format;

use super::Descriptor;
use super::{Filter, FilterMut};

#[test]
fn set_4d_nchw() {
    let mut desc: Descriptor<f32> = Descriptor::new().unwrap();
    desc.set_4d(Format::NCHW, 2, 3, 5, 7).unwrap();
    assert_eq!(desc.len(), 2 * 3 * 5 * 7);
}

#[test]
fn set_4d_nhwc() {
    let mut desc: Descriptor<f32> = Descriptor::new().unwrap();
    desc.set_4d(Format::NHWC, 2, 3, 5, 7).unwrap();
    assert_eq!(desc.len(), 2 * 3 * 5 * 7);
}

#[test]
fn filter() {
    let mut desc: Descriptor<f32> = Descriptor::new().unwrap();
    desc.set_4d(Format::NCHW, 2, 3, 5, 7).unwrap();
    let mem = memory::Memory::new(desc.len()).unwrap();
    let filter = Filter::new(&desc, &mem);
    assert_eq!(filter.desc().as_ptr(), desc.as_ptr());
    assert_eq!(filter.mem().as_ptr(), mem.as_ptr());
}

#[test]
#[should_panic]
fn filter_invalid() {
    let mut desc: Descriptor<f32> = Descriptor::new().unwrap();
    desc.set_4d(Format::NCHW, 2, 3, 5, 7).unwrap();
    let mem = memory::Memory::new(desc.len() - 1).unwrap();
    Filter::new(&desc, &mem);
}

#[test]
fn filter_mut() {
    let mut desc: Descriptor<f32> = Descriptor::new().unwrap();
    desc.set_4d(Format::NCHW, 2, 3, 5, 7).unwrap();
    let mut mem = memory::Memory::new(desc.len()).unwrap();
    let mem_ptr = mem.as_ptr();
    let mem_mut_ptr = mem.as_mut_ptr();
    let mut filter = FilterMut::new(&desc, &mut mem);
    assert_eq!(filter.desc().as_ptr(), desc.as_ptr());
    assert_eq!(filter.mem().as_ptr(), mem_ptr);
    assert_eq!(filter.mem_mut().as_mut_ptr(), mem_mut_ptr);
}

#[test]
#[should_panic]
fn filter_mut_invalid() {
    let mut desc: Descriptor<f32> = Descriptor::new().unwrap();
    desc.set_4d(Format::NCHW, 2, 3, 5, 7).unwrap();
    let mut mem = memory::Memory::new(desc.len() - 1).unwrap();
    FilterMut::new(&desc, &mut mem);
}
