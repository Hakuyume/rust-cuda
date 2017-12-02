extern crate rand;
use self::rand::Rng;

use cuda::memory;

use context;
use tensor;

use super::Algorithm;
use super::Mode;
use super::forward;
use super::backward;

const N: usize = 2;
const C: usize = 3;
const H: usize = 5;
const W: usize = 7;


#[test]
fn forward_channel() {
    let mut context = context::Context::new().unwrap();

    let mut desc: tensor::Descriptor<f32> = tensor::Descriptor::new().unwrap();
    desc.set_4d(tensor::Format::NCHW, N, C, H, W).unwrap();

    let x: Vec<_> = {
        let mut rng = rand::thread_rng();
        (0..desc.len()).map(|_| rng.gen()).collect()
    };

    let expected_y = {
        let mut y: Vec<f32> = x.iter().map(|x| x.exp()).collect();
        for n in 0..N {
            for k in 0..(H * W) {
                let mut sum = 0.;
                for c in 0..C {
                    sum += y[(n * C + c) * (H * W) + k];
                }
                for c in 0..C {
                    y[(n * C + c) * (H * W) + k] /= sum;
                }
            }
        }
        y
    };

    let mut dev_x = memory::Memory::new(desc.len()).unwrap();
    memory::memcpy(&mut dev_x, &x).unwrap();

    {
        let mut dev_y = memory::Memory::new(desc.len()).unwrap();
        forward(&mut context,
                Algorithm::Accurate,
                Mode::Channel,
                1.,
                tensor::Tensor::new(&desc, &dev_x),
                0.,
                tensor::TensorMut::new(&desc, &mut dev_y))
                .unwrap();
        let mut y = vec![0.; desc.len()];
        memory::memcpy(&mut y, &dev_y);

        for k in 0..desc.len() {
            assert_eq!(y[k], expected_y[k]);
        }
    }
}
