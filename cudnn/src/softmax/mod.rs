use cudnn_sys;
use cudnn_sys::c_void;

use scalar;
use Result;
use context;
use tensor;

use cuda::memory::{Repr, ReprMut};

mod algorithm;
pub use self::algorithm::Algorithm;

mod mode;
pub use self::mode::Mode;

pub unsafe fn forward<T, S, X, Y>(context: &mut context::Context,
                                  algo: Algorithm,
                                  mode: Mode,
                                  alpha: S,
                                  x: (&tensor::Descriptor<T>, &X),
                                  beta: S,
                                  y: (&tensor::Descriptor<T>, &mut Y))
                                  -> Result<()>
    where T: scalar::Scalar + scalar::Scale<Scale = S>,
          X: Repr<T>,
          Y: ReprMut<T>
{
    try_call!(cudnn_sys::cudnnSoftmaxForward(context.as_mut_ptr(),
                                             algo.into(),
                                             mode.into(),
                                             &alpha as *const S as *const c_void,
                                             x.0.as_ptr(),
                                             x.1.as_ptr() as *const c_void,
                                             &beta as *const S as *const c_void,
                                             y.0.as_ptr(),
                                             y.1.as_mut_ptr() as *mut c_void));
    Ok(())
}

pub unsafe fn backward<T, S, Y, DY, DX>(context: &mut context::Context,
                                        algo: Algorithm,
                                        mode: Mode,
                                        alpha: S,
                                        y: (&tensor::Descriptor<T>, &Y),
                                        dy: Option<(&tensor::Descriptor<T>, &DY)>,
                                        beta: S,
                                        dx: (&tensor::Descriptor<T>, &mut DX))
                                        -> Result<()>
    where T: scalar::Scalar + scalar::Scale<Scale = S>,
          Y: Repr<T>,
          DY: Repr<T>,
          DX: ReprMut<T>
{
    let dy_ptr = match dy {
        Some(dy) => (dy.0.as_ptr(), dy.1.as_ptr()),
        None => (dx.0.as_ptr(), dx.1.as_ptr()),
    };
    try_call!(cudnn_sys::cudnnSoftmaxBackward(context.as_mut_ptr(),
                                              algo.into(),
                                              mode.into(),
                                              &alpha as *const S as *const c_void,
                                              y.0.as_ptr(),
                                              y.1.as_ptr() as *const c_void,
                                              dy_ptr.0,
                                              dy_ptr.1 as *const c_void,
                                              &beta as *const S as *const c_void,
                                              dx.0.as_ptr(),
                                              dx.1.as_mut_ptr() as *mut c_void));
    Ok(())
}

#[cfg(test)]
mod tests;
