use Error;
use super::View;

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
        Ok(_) => panic!("allocation of a huge memory returned successfully"),
        Err(e) => panic!("{:?}", e),
    }
}

#[test]
#[should_panic(expected = "malloc: 64")]
fn malloc_hook() {
    super::set_malloc_hook(|_, size| panic!("malloc: {}", size));
    let _m = super::Memory::<f32>::new(16).unwrap();
    unreachable!();
}

#[test]
#[should_panic(expected = "free: 64")]
fn free_hook() {
    {
        let _m = super::Memory::<f32>::new(16).unwrap();
        super::set_free_hook(|_, size| panic!("free: {}", size));
    }
    unreachable!();
}
