extern crate rand;
use self::rand::Rng;

use memory::Memory;
use super::Repr;

#[test]
fn memcpy() {
    let src: Vec<f32> = {
        let mut rng = rand::thread_rng();
        (0..256).map(|_| rng.gen()).collect()
    };

    let mut dev = Memory::new(src.len()).unwrap();
    super::memcpy(&mut dev, &src).unwrap();

    let mut dst = vec![0.; dev.len()];
    super::memcpy(&mut dst, &dev).unwrap();
    assert_eq!(&dst, &src);
}

#[test]
#[should_panic]
fn memcpy_host_to_device_invalid() {
    let host = vec![0.; 128];
    let mut dev = Memory::new(256).unwrap();
    super::memcpy(&mut dev, &host).unwrap();
}

#[test]
#[should_panic]
fn memcpy_device_to_host_invalid() {
    let mut host = vec![0.; 256];
    let dev = Memory::new(128).unwrap();
    super::memcpy(&mut host, &dev).unwrap();
}
