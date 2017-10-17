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

pub fn forward<'a, T: scalar::Float>(context: &mut context::Context,
                                     algo: Algorithm,
                                     mode: Mode,
                                     alpha: T,
                                     x: tensor::Tensor<'a, T>,
                                     beta: T,
                                     mut y: tensor::TensorMut<'a, T>)
                                     -> Result<()> {
    let scales: &[T::Scale] = &[alpha.into(), beta.into()];
    unsafe {
        try_call!(cudnn_sys::cudnnSoftmaxForward(context.as_mut_ptr(),
                                                 algo.into(),
                                                 mode.into(),
                                                 &scales[0] as *const T::Scale as *const c_void,
                                                 x.desc().as_ptr(),
                                                 x.mem().as_ptr() as *const c_void,
                                                 &scales[1] as *const T::Scale as *const c_void,
                                                 y.desc().as_ptr(),
                                                 y.mem_mut().as_mut_ptr() as *mut c_void))
    }
    Ok(())
}
