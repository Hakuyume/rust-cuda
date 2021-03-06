use std::os::raw::c_void;

use cuda::memory::{Repr, ReprMut};
use cudnn_sys;

use Result;
use scalar;
use context;
use tensor;

mod mode;
pub use self::mode::Mode;

mod descriptor;
pub use self::descriptor::Descriptor;

use helper::MemoryDescriptor;

pub fn forward<T, S, Src, Dest>(context: &mut context::Context,
                                activation_desc: &Descriptor,
                                alpha: &S,
                                src: Option<(&tensor::Descriptor<T>, &Src)>,
                                beta: &S,
                                dest: (&tensor::Descriptor<T>, &mut Dest))
                                -> Result<()>
    where T: scalar::Scale<Scale = S>,
          Src: Repr<T>,
          Dest: ReprMut<T>
{
    if let Some(ref src) = src {
        src.0.check_memory(src.1)?;
    }
    dest.0.check_memory(dest.1)?;

    let src_ptr = match src {
        Some(src) => (src.0.as_ptr(), src.1.as_ptr()),
        None => (dest.0.as_ptr(), dest.1.as_ptr()),
    };
    unsafe {
        try_call!(cudnn_sys::cudnnActivationForward(context.as_mut_ptr(),
                                                    activation_desc.as_ptr(),
                                                    alpha as *const S as *const c_void,
                                                    src_ptr.0,
                                                    src_ptr.1 as *const c_void,
                                                    beta as *const S as *const c_void,
                                                    dest.0.as_ptr(),
                                                    dest.1.as_mut_ptr() as *mut c_void))
    }
    Ok(())
}
