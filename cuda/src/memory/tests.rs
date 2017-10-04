use super::*;

use Error;

#[test]
fn malloc() {
    match Memory::<f32>::new(16) {
        Ok(m) => assert_eq!(m.len(), 16),
        Err(e) => panic!("{:?}", e),
    }
}

#[test]
fn malloc_huge() {
    match Memory::<f32>::new(1 << 48) {
        Err(Error::MemoryAllocation) => (),
        Ok(_) => panic!("allocation of a huge memory returned succeccfully"),
        Err(e) => panic!("{:?}", e),
    }
}