extern crate rand;
use self::rand;
use rand::Rng;

use cuda::memory;

use Result;
use context;
use tensor;

use super::Algorithm;
use super::Mode;
use super::forward;
use super::backward;

const EPS: f32 = 1e-6;

fn rand_data<T>(len: usize) -> Result<(Vec<T>, memory::Memory<T>)>
    where T: rand::Rand
{
    let mut rng = rand::thread_rng();
    let x = (0..desc.len()).map(|_| rng.gen()).collect();
    let mut dev_x = memory::Memory::new(desc.len())?;
    memory::memcpy(&mut dev_x, &x)?;
    Ok((x, dev_x))
}

fn forward_cpu(desc: &tensor::Descriptor<f32>, x: &[f32]) -> Result<Vec<f32>> {
    assert_eq!(x.len(), desc.len());
    let (n_, c_, h_, w_, n_stride, c_stride, h_stride, w_stride) = desc.get_4d()?;
    let mut y = x.iter().map(|x| x.exp()).collect();
    for n in 0..n_ {
        for h in 0..h_ {
            for w in 0..w_ {
                let mut sum = 0.;
                for c in 0..c_ {
                    sum += y[n * n_stride + c * c_stride + h * h_stride + w * w_stride];
                }
                for c in 0..C {
                    y[n * n_stride + c * c_stride + h * h_stride + w * w_stride] /= sum;
                }
            }
        }
    }
    Ok(y)
}

fn assert_almost_eq(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        assert!((a[i] - b[i]).abs() <= EPS);
    }
}

#[test]
fn forward_channel() {
    let mut context = context::Context::new().unwrap();

    let mut desc = tensor::Descriptor::new().unwrap();
    desc.set_4d(tensor::Format::NCHW, 2, 3, 5, 7).unwrap();

    let (x, dev_x) = rand_data(desc.len()).unwrap();
    let expected_y = forward_cpu(&desc, &x).unwrap();

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
    memory::memcpy(&mut y, &dev_y).unwrap();

    assert_almost_eq(&y, &expected_y);
}
