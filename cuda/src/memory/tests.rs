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
    let host_src: Vec<f32> = {
        let mut rng = rand::thread_rng();
        (0..256).map(|_| rng.gen()).collect()
    };

    let mut device = super::Memory::new(host_src.len()).unwrap();
    super::memcpy(&mut device, &host_src).unwrap();

    let mut host_dst = vec![0.; device.len()];
    super::memcpy(&mut host_dst, &device).unwrap();
    assert_eq!(&host_dst, &host_src);
}

#[test]
#[should_panic]
fn memcpy_host_to_device_invalid() {
    let host = vec![0.; 128];
    let mut device = super::Memory::new(256).unwrap();
    super::memcpy(&mut device, &host).unwrap();
}

#[test]
#[should_panic]
fn memcpy_device_to_host_invalid() {
    let device = super::Memory::new(128).unwrap();
    let mut host = vec![0.; 256];
    super::memcpy(&mut host, &device).unwrap();
}
