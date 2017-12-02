use std::fmt;

extern crate num_traits;
extern crate rand;

use self::rand::Rng;

use cuda;
use cuda::memory;

use Result;
use scalar;
use context;
use tensor;

use super::Algorithm;
use super::Mode;
use super::forward;
use super::backward;

fn rand_data<T>(len: usize) -> cuda::Result<(Vec<T>, memory::Memory<T>)>
    where T: rand::Rand
{
    let mut rng = rand::thread_rng();
    let x: Vec<_> = (0..len).map(|_| rng.gen()).collect();
    let mut dev_x = memory::Memory::new(len)?;
    memory::memcpy(&mut dev_x, &x)?;
    Ok((x, dev_x))
}

fn forward_cpu<T>(desc: &tensor::Descriptor<T>, x: &[T]) -> Result<Vec<T>>
    where T: scalar::Scalar + num_traits::float::Float + num_traits::NumAssignOps
{
    assert_eq!(x.len(), desc.len());
    let (n_, c_, h_, w_, n_stride, c_stride, h_stride, w_stride) = desc.get_4d()?;
    let mut y: Vec<_> = x.iter().map(|x| x.exp()).collect();
    for n in 0..n_ {
        for h in 0..h_ {
            for w in 0..w_ {
                let mut sum = T::zero();
                for c in 0..c_ {
                    sum += y[n * n_stride + c * c_stride + h * h_stride + w * w_stride];
                }
                for c in 0..c_ {
                    y[n * n_stride + c * c_stride + h * h_stride + w * w_stride] /= sum;
                }
            }
        }
    }
    Ok(y)
}

fn assert_almost_eq<T>(a: &[T], b: &[T])
    where T: fmt::Display + num_traits::float::Float + From<f32>
{
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        if (a[i] - b[i]).abs() > (1e-6).into() {
            panic!("{} th elements differ: {} != {}", i, a[i], b[i]);
        }
    }
}

#[test]
fn forward_channel() {
    let mut context = context::Context::new().unwrap();

    let mut desc = tensor::Descriptor::new().unwrap();
    desc.set_4d(tensor::Format::NCHW, 2, 3, 5, 7).unwrap();

    let (x, dev_x) = rand_data(desc.len()).unwrap();
    let expected_y = forward_cpu(&desc, &x).unwrap();

    for algo in &[Algorithm::Accurate, Algorithm::Fast, Algorithm::Log] {
        let mut dev_y = memory::Memory::new(desc.len()).unwrap();
        forward(&mut context,
                *algo,
                Mode::Channel,
                1.,
                tensor::Tensor::new(&desc, &dev_x),
                0.,
                tensor::TensorMut::new(&desc, &mut dev_y))
                .unwrap();
        let mut y = vec![0.; desc.len()];
        memory::memcpy(&mut y, &dev_y).unwrap();

        let expected_y: Vec<_> = match *algo {
            Algorithm::Log => expected_y.iter().map(|y| y.ln()).collect(),
            _ => expected_y.clone(),
        };
        assert_almost_eq(&y, &expected_y);
    }
}
