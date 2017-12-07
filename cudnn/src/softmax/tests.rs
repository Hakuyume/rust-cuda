use std::iter;
use std::mem;

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

use self::rand::distributions::IndependentSample;

fn random_data<T>(len: usize) -> cuda::Result<(Vec<T>, memory::Memory<T>)>
    where T: rand::distributions::range::SampleRange + num_traits::Float
{
    let mut rng = rand::thread_rng();
    let dist = rand::distributions::Range::new(-T::one(), T::one());
    let x: Vec<_> = (0..len).map(|_| dist.ind_sample(&mut rng)).collect();
    let mut x_dev = memory::Memory::new(len)?;
    memory::memcpy(&mut x_dev, &x)?;
    Ok((x, x_dev))
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
    where T: num_traits::Float + From<f32>
{
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        assert!((a[i] - b[i]).abs() < (1e-6).into());
    }
}

fn test_forward(algo: Algorithm, mode: Mode) {
    let mut context = context::Context::new().unwrap();

    let mut desc = tensor::Descriptor::<f32>::new().unwrap();
    desc.set_4d(tensor::Format::NCHW, 2, 3, 5, 7).unwrap();
    let len = desc.get_size_in_bytes().unwrap() / mem::size_of::<f32>();

    let (x, x_dev) = random_data(len).unwrap();
    let (mut y, mut y_dev) = random_data(len).unwrap();
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
            (&desc, &x_dev),
            beta,
            (&desc, &mut y_dev))
            .unwrap();
    memory::memcpy(&mut y, &y_dev).unwrap();

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
