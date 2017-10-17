use cudnn_sys;
use cudnn_sys::c_void;

use scalar;
use Result;
use context;
use tensor;

mod mode;
pub use self::mode::Mode;

mod descriptor;
pub use self::descriptor::Descriptor;

pub fn forward<'a, T: scalar::Float>(context: &mut context::Context,
                                     activation_desc: &Descriptor<T>,
                                     alpha: T,
                                     src: Option<tensor::Tensor<'a, T>>,
                                     beta: T,
                                     mut dest: tensor::TensorMut<'a, T>)
                                     -> Result<()> {
    let scales: &[T::Scale] = &[alpha.into(), beta.into()];
    let (src_desc, src) = match src {
        Some(ref src) => (src.desc().as_ptr(), src.mem().as_ptr()),
        None => (dest.desc().as_ptr(), dest.mem().as_ptr()),
    };

    unsafe {
        try_call!(cudnn_sys::cudnnActivationForward(context.as_mut_ptr(),
                                                    activation_desc.as_ptr(),
                                                    &scales[0] as *const T::Scale as *const c_void,
                                                    src_desc,
                                                    src as *const c_void,
                                                    &scales[1] as *const T::Scale as *const c_void,
                                                    dest.desc().as_ptr(),
                                                    dest.mem_mut().as_mut_ptr() as *mut c_void))
    }
    Ok(())
}
