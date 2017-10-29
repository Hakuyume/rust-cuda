extern crate rand;
use self::rand::Rng;

use super::super::Memory;
use super::View;

#[test]
fn memcpy() {
    let host_src: Vec<f32> = {
        let mut rng = rand::thread_rng();
        (0..256).map(|_| rng.gen()).collect()
    };

    let mut device = Memory::new(host_src.len()).unwrap();
    super::memcpy(&mut device, &host_src).unwrap();

    let mut host_dst = vec![0.; device.len()];
    super::memcpy(&mut host_dst, &device).unwrap();
    assert_eq!(&host_dst, &host_src);
}

#[test]
#[should_panic]
fn memcpy_host_to_device_invalid() {
    let host = vec![0.; 128];
    if let Ok(mut device) = Memory::new(256) {
        super::memcpy(&mut device, &host).unwrap();
    }
}

#[test]
#[should_panic]
fn memcpy_device_to_host_invalid() {
    let mut host = vec![0.; 256];
    if let Ok(device) = Memory::new(128) {
        super::memcpy(&mut host, &device).unwrap();
    }
}
