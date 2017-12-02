use cuda::memory::{Repr, ReprMut};

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

pub fn forward<'a, T, S>(context: &mut context::Context,
                         activation_desc: &Descriptor<T>,
                         alpha: S,
                         src: Option<tensor::Tensor<'a, T>>,
                         beta: S,
                         mut dest: tensor::TensorMut<'a, T>)
                         -> Result<()>
    where T: scalar::Scalar + scalar::Scale<Scale = S>
{
    let (src_desc, src) = match src {
        Some(src) => (src.desc().as_ptr(), src.mem().as_ptr()),
        None => (dest.desc().as_ptr(), dest.mem().as_ptr()),
    };
    unsafe {
        try_call!(cudnn_sys::cudnnActivationForward(context.as_mut_ptr(),
                                                    activation_desc.as_ptr(),
                                                    &alpha as *const S as *const c_void,
                                                    src_desc,
                                                    src as *const c_void,
                                                    &beta as *const S as *const c_void,
                                                    dest.desc().as_ptr(),
                                                    dest.mem_mut().as_mut_ptr() as *mut c_void))
    }
    Ok(())
}
