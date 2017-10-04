use super::*;

use Error;

#[test]
fn alloc() {
    match Memory::<f32>::new(16) {
        Ok(m) => assert_eq!(m.len(), 16),
        Err(e) => panic!("{:?}", e),
    }
}

#[test]
fn alloc_huge() {
    match Memory::<f32>::new(1 << 48) {
        Ok(m) => assert_eq!(m.len(), 1 << 48),
        Err(Error::MemoryAllocation) => (),
        Err(e) => panic!("{:?}", e),
    }
}
