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

pub fn forward<'a, T, X, Y>(context: &context::Context,
                            algo: Algorithm,
                            mode: Mode,
                            alpha: T,
                            x: tensor::Tensor<'a, T>,
                            beta: T,
                            y: tensor::TensorMut<'a, T>)
                            -> Result<()>
    where T: scalar::Float
{
    let params: &[T::Scale] = &[alpha.into(), beta.into()];
    unsafe {
        try_call!(cudnn_sys::cudnnSoftmaxForward(context.as_raw(),
                                                 algo.as_raw(),
                                                 mode.as_raw(),
                                                 &params[0] as *const T::Scale as *const c_void,
                                                 x.desc.as_raw(),
                                                 x.mem.as_ptr() as *const c_void,
                                                 &params[1] as *const T::Scale as *const c_void,
                                                 y.desc.as_raw(),
                                                 y.mem.as_mut_ptr() as *mut c_void))
    }
    Ok(())
}
