use std::os::raw::c_void;

use cuda::memory::ReprMut;
use cudnn_sys;

use Result;
use scalar;
use context;

mod format;
pub use self::format::Format;

mod descriptor;
pub use self::descriptor::Descriptor;

use misc::MemoryDescriptor;

pub fn scale<T, S, Y>(context: &mut context::Context,
                      y: (&Descriptor<T>, &mut Y),
                      alpha: S)
                      -> Result<()>
    where T: scalar::Scalar + scalar::Scale<Scale = S>,
          Y: ReprMut<T>
{
    y.0.check_memory(y.1)?;
    unsafe {
        try_call!(cudnn_sys::cudnnScaleTensor(context.as_mut_ptr(),
                                              y.0.as_ptr(),
                                              y.1.as_mut_ptr() as *mut c_void,
                                              &alpha as *const S as *const c_void))
    }
    Ok(())
}

#[cfg(test)]
mod tests;
