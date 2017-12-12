use Error;
use memory::Repr;

#[test]
fn malloc() {
    let m = super::Memory::<f32>::new(16).unwrap();
    assert_eq!(m.len(), 16);
}

#[test]
fn malloc_huge() {
    match super::Memory::<f32>::new(1 << 48) {
        Err(Error::MemoryAllocation) => (),
        Ok(_) => panic!("Allocation of a huge memory should fail"),
        Err(e) => panic!("{:?}", e),
    }
}
