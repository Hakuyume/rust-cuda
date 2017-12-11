use Error;
use super::Repr;

#[test]
fn malloc() {
    match super::Memory::<f32>::new(16) {
        Ok(m) => assert_eq!(m.len(), 16),
        Err(e) => panic!("{:?}", e),
    }
}

#[test]
fn malloc_huge() {
    match super::Memory::<f32>::new(1 << 48) {
        Err(Error::MemoryAllocation) => (),
        Ok(_) => panic!("Allocation of a huge memory should fail"),
        Err(e) => panic!("{:?}", e),
    }
}
