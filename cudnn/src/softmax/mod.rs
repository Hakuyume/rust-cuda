use std::os::raw::c_void;

use cuda::memory::{Repr, ReprMut};
use cudnn_sys;

use Result;
use scalar;
use context;
use tensor;

mod algorithm;
pub use self::algorithm::Algorithm;

mod mode;
pub use self::mode::Mode;

use misc::MemoryDescriptor;

pub fn forward<T, S, X, Y>(context: &mut context::Context,
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
    x.0.check_memory(x.1)?;
    y.0.check_memory(y.1)?;
    unsafe {
        try_call!(cudnn_sys::cudnnSoftmaxForward(context.as_mut_ptr(),
                                                 algo.into(),
                                                 mode.into(),
                                                 &alpha as *const S as *const c_void,
                                                 x.0.as_ptr(),
                                                 x.1.as_ptr() as *const c_void,
                                                 &beta as *const S as *const c_void,
                                                 y.0.as_ptr(),
                                                 y.1.as_mut_ptr() as *mut c_void))
    }
    Ok(())
}

pub fn backward<T, S, Y, Dy, Dx>(context: &mut context::Context,
                                 algo: Algorithm,
                                 mode: Mode,
                                 alpha: S,
                                 y: (&tensor::Descriptor<T>, &Y),
                                 dy: Option<(&tensor::Descriptor<T>, &Dy)>,
                                 beta: S,
                                 dx: (&tensor::Descriptor<T>, &mut Dx))
                                 -> Result<()>
    where T: scalar::Scalar + scalar::Scale<Scale = S>,
          Y: Repr<T>,
          Dy: Repr<T>,
          Dx: ReprMut<T>
{
    y.0.check_memory(y.1)?;
    if let Some(ref dy) = dy {
        dy.0.check_memory(dy.1)?;
    }
    dx.0.check_memory(dx.1)?;

    let dy_ptr = match dy {
        Some(dy) => (dy.0.as_ptr(), dy.1.as_ptr()),
        None => (dx.0.as_ptr(), dx.1.as_ptr()),
    };
    unsafe {
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
                                                  dx.1.as_mut_ptr() as *mut c_void))
    }
    Ok(())
}
