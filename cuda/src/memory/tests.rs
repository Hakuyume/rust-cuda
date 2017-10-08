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
#[should_panic(expected = "malloc: 64")]
fn malloc_hook() {
    super::set_malloc_hook(|_, size| panic!("malloc: {}", size));
    super::Memory::<f32>::new(16).unwrap();
    unreachable!();
}

#[test]
#[should_panic(expected = "free: 64")]
fn free_hook() {
    {
        super::Memory::<f32>::new(16).unwrap();
        super::set_free_hook(|_, size| panic!("free: {}", size));
    }
    unreachable!();
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
    if let Ok(mut device) = super::Memory::new(256) {
        super::memcpy(&mut device, &host).unwrap();
    }
}

#[test]
#[should_panic]
fn memcpy_device_to_host_invalid() {
    let mut host = vec![0.; 256];
    if let Ok(device) = super::Memory::new(128) {
        super::memcpy(&mut host, &device).unwrap();
    }
}
