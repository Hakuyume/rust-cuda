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

pub unsafe fn forward<T, S, SRC, DEST>(context: &mut context::Context,
                                       activation_desc: &Descriptor<T>,
                                       alpha: S,
                                       src: Option<(&tensor::Descriptor<T>, &SRC)>,
                                       beta: S,
                                       dest: (&tensor::Descriptor<T>, &mut DEST))
                                       -> Result<()>
    where T: scalar::Scalar + scalar::Scale<Scale = S>,
          SRC: Repr<T>,
          DEST: ReprMut<T>
{
    let src_ptr = match src {
        Some(src) => (src.0.as_ptr(), src.1.as_ptr()),
        None => (dest.0.as_ptr(), dest.1.as_ptr()),
    };
    try_call!(cudnn_sys::cudnnActivationForward(context.as_mut_ptr(),
                                                activation_desc.as_ptr(),
                                                &alpha as *const S as *const c_void,
                                                src_ptr.0,
                                                src_ptr.1 as *const c_void,
                                                &beta as *const S as *const c_void,
                                                dest.0.as_ptr(),
                                                dest.1.as_mut_ptr() as *mut c_void));
    Ok(())
}
