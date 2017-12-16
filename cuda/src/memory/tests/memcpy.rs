extern crate rand;

use super::Array;
use super::super::memcpy;

use self::rand::Rng;

#[test]
fn memcpy_random() {
    let src: Vec<_> = {
        let mut rng = rand::thread_rng();
        (0..16).map(|_| rng.gen()).collect()
    };

    let mut dev = Array::new(src.len()).unwrap();
    memcpy(&mut dev, &src).unwrap();

    let mut dst = vec![0; dev.len()];
    memcpy(&mut dst, &dev).unwrap();
    assert_eq!(&dst, &src);
}

#[test]
#[should_panic(expected = "assertion failed")]
fn memcpy_host_to_device_invalid() {
    let host = vec![0; 16];
    let mut dev = Array::new(32).unwrap();
    memcpy(&mut dev, &host).unwrap();
}

#[test]
#[should_panic(expected = "assertion failed")]
fn memcpy_device_to_host_invalid() {
    let mut host = vec![0; 32];
    let dev = Array::new(16).unwrap();
    memcpy(&mut host, &dev).unwrap();
}
