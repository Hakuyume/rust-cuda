use cuda::memory::{Repr, ReprMut};

use cudnn_sys;
use cudnn_sys::c_void;

use scalar;
use Result;
use context;
use tensor;

mod algorithm;
pub use self::algorithm::Algorithm;

mod mode;
pub use self::mode::Mode;

pub fn forward<'a, T, S>(context: &mut context::Context,
                         algo: Algorithm,
                         mode: Mode,
                         alpha: S,
                         x: tensor::Tensor<'a, T>,
                         beta: S,
                         mut y: tensor::TensorMut<'a, T>)
                         -> Result<()>
    where T: scalar::Scalar + scalar::Scale<Scale = S>
{
    unsafe {
        try_call!(cudnn_sys::cudnnSoftmaxForward(context.as_mut_ptr(),
                                                 algo.into(),
                                                 mode.into(),
                                                 &alpha as *const T::Scale as *const c_void,
                                                 x.desc().as_ptr(),
                                                 x.mem().as_ptr() as *const c_void,
                                                 &beta as *const T::Scale as *const c_void,
                                                 y.desc().as_ptr(),
                                                 y.mem_mut().as_mut_ptr() as *mut c_void))
    }
    Ok(())
}

pub fn backward<'a, T, S>(context: &mut context::Context,
                          algo: Algorithm,
                          mode: Mode,
                          alpha: S,
                          y: tensor::Tensor<'a, T>,
                          dy: Option<tensor::Tensor<'a, T>>,
                          beta: S,
                          mut dx: tensor::TensorMut<'a, T>)
                          -> Result<()>
    where T: scalar::Scalar + scalar::Scale<Scale = S>
{
    let (dy_desc, dy) = match dy {
        Some(dy) => (dy.desc().as_ptr(), dy.mem().as_ptr()),
        None => (dx.desc().as_ptr(), dx.mem().as_ptr()),
    };
    unsafe {
        try_call!(cudnn_sys::cudnnSoftmaxBackward(context.as_mut_ptr(),
                                                  algo.into(),
                                                  mode.into(),
                                                  &alpha as *const T::Scale as *const c_void,
                                                  y.desc().as_ptr(),
                                                  y.mem().as_ptr() as *const c_void,
                                                  dy_desc,
                                                  dy as *const c_void,
                                                  &beta as *const T::Scale as *const c_void,
                                                  dx.desc().as_ptr(),
                                                  dx.mem_mut().as_mut_ptr() as *mut c_void))
    }
    Ok(())
}

#[cfg(test)]
mod tests;
