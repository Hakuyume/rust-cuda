use std::fmt;
use std::iter;

extern crate num_traits;
extern crate rand;

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

use self::rand::distributions::IndependentSample;

fn random_data<T>(len: usize) -> cuda::Result<(Vec<T>, memory::Memory<T>)>
    where T: rand::distributions::range::SampleRange + num_traits::Float
{
    let mut rng = rand::thread_rng();
    let dist = rand::distributions::Range::new(-T::one(), T::one());
    let x: Vec<_> = (0..len).map(|_| dist.ind_sample(&mut rng)).collect();
    let mut dev_x = memory::Memory::new(len)?;
    memory::memcpy(&mut dev_x, &x)?;
    Ok((x, dev_x))
}

fn random_coeff<T>() -> T
    where T: rand::distributions::range::SampleRange + num_traits::Float
{
    let mut rng = rand::thread_rng();
    let dist = rand::distributions::Range::new(T::zero(), T::one());
    dist.ind_sample(&mut rng)
}

fn forward_cpu<T>(algo: Algorithm,
                  mode: Mode,
                  desc: &tensor::Descriptor<T>,
                  x: &[T])
                  -> Result<Vec<T>>
    where T: scalar::Scalar + num_traits::Float + num_traits::NumAssignOps + iter::Sum
{
    assert_eq!(x.len(), desc.len());
    let (n_, c_, h_, w_, n_stride, c_stride, h_stride, w_stride) = desc.get_4d()?;
    let mut y: Vec<_> = x.iter().map(|x| x.exp()).collect();
    for n in 0..n_ {
        match mode {
            Mode::Channel => {
                for h in 0..h_ {
                    for w in 0..w_ {
                        let sum = (0..c_)
                            .map(|c| y[n * n_stride + c * c_stride + h * h_stride + w * w_stride])
                            .sum();
                        for c in 0..c_ {
                            y[n * n_stride + c * c_stride + h * h_stride + w * w_stride] /= sum;
                        }
                    }
                }
            }
            Mode::Instance => {
                let sum = (0..n_stride).map(|i| y[n * n_stride + i]).sum();
                for i in 0..n_stride {
                    y[n * n_stride + i] /= sum;
                }
            }
        }
    }
    if algo == Algorithm::Log {
        for y in y.iter_mut() {
            *y = y.ln();
        }
    }
    Ok(y)
}

fn assert_almost_eq<T>(a: &[T], b: &[T])
    where T: fmt::Display + num_traits::Float + From<f32>
{
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        if (a[i] - b[i]).abs() > (1e-6).into() {
            panic!("{} th elements differ: {} != {}", i, a[i], b[i]);
        }
    }
}

fn test_forward(algo: Algorithm, mode: Mode) {
    let mut context = context::Context::new().unwrap();

    let desc = tensor::Descriptor::new_4d(tensor::Format::NCHW, 2, 3, 5, 7).unwrap();

    let (x, dev_x) = random_data::<f32>(desc.len()).unwrap();
    let (mut y, mut dev_y) = random_data(desc.len()).unwrap();
    let (alpha, beta) = (random_coeff(), random_coeff());

    let expected: Vec<_> = forward_cpu(algo, mode, &desc, &x)
        .unwrap()
        .into_iter()
        .zip(&y)
        .map(|(x, y)| x * alpha + y * beta)
        .collect();

    forward(&mut context,
            algo,
            mode,
            alpha,
            tensor::Tensor::new(&desc, &dev_x),
            beta,
            tensor::TensorMut::new(&desc, &mut dev_y))
            .unwrap();
    memory::memcpy(&mut y, &dev_y).unwrap();

    assert_almost_eq(&y, &expected);
}

#[test]
fn forward_accurate_channel() {
    test_forward(Algorithm::Accurate, Mode::Channel);
}

#[test]
fn forward_fast_channel() {
    test_forward(Algorithm::Fast, Mode::Channel);
}

#[test]
fn forward_log_channel() {
    test_forward(Algorithm::Log, Mode::Channel);
}

#[test]
fn forward_accurate_instance() {
    test_forward(Algorithm::Accurate, Mode::Instance);
}

#[test]
fn forward_fast_instance() {
    test_forward(Algorithm::Fast, Mode::Instance);
}

#[test]
fn forward_log_instance() {
    test_forward(Algorithm::Log, Mode::Instance);
}
