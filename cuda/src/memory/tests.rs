extern crate rand;
use self::rand::Rng;

use Error;

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
fn memcpy() {
    let mut host_src = vec![0.; 256];
    {
        let mut rng = rand::thread_rng();
        for x in host_src.iter_mut() {
            *x = rng.gen::<f32>();
        }
    }

    let mut device = super::Memory::new(host_src.len()).unwrap();
    super::memcpy(&mut device, &host_src).unwrap();

    let mut host_dst = vec![0.; device.len()];
    super::memcpy(&mut host_dst, &device).unwrap();
    assert_eq!(&host_dst, &host_src);
}
