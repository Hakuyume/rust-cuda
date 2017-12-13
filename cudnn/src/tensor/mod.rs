use std::os::raw::c_void;

use cuda::memory::{Repr, ReprMut};
use cudnn_sys;

use Result;
use scalar;
use context;

mod format;
pub use self::format::Format;

mod descriptor;
pub use self::descriptor::Descriptor;

use misc::MemoryDescriptor;

pub fn add<T, S, A, C>(context: &mut context::Context,
                       alpha: &S,
                       a: (&Descriptor<T>, &A),
                       beta: &S,
                       c: (&Descriptor<T>, &mut C))
                       -> Result<()>
    where T: scalar::Scalar + scalar::Scale<Scale = S>,
          A: Repr<T>,
          C: ReprMut<T>
{
    a.0.check_memory(a.1)?;
    c.0.check_memory(c.1)?;
    unsafe {
        try_call!(cudnn_sys::cudnnAddTensor(context.as_mut_ptr(),
                                            alpha as *const S as *const c_void,
                                            a.0.as_ptr(),
                                            a.1.as_ptr() as *const c_void,
                                            beta as *const S as *const c_void,
                                            c.0.as_ptr(),
                                            c.1.as_mut_ptr() as *mut c_void))
    }
    Ok(())
}

pub fn set<T, Y>(context: &mut context::Context,
                 y: (&Descriptor<T>, &mut Y),
                 value: &T)
                 -> Result<()>
    where T: scalar::Scalar,
          Y: ReprMut<T>
{
    y.0.check_memory(y.1)?;
    unsafe {
        try_call!(cudnn_sys::cudnnSetTensor(context.as_mut_ptr(),
                                            y.0.as_ptr(),
                                            y.1.as_mut_ptr() as *mut c_void,
                                            value as *const T as *const c_void))
    }
    Ok(())
}

pub fn scale<T, S, Y>(context: &mut context::Context,
                      y: (&Descriptor<T>, &mut Y),
                      alpha: &S)
                      -> Result<()>
    where T: scalar::Scalar + scalar::Scale<Scale = S>,
          Y: ReprMut<T>
{
    y.0.check_memory(y.1)?;
    unsafe {
        try_call!(cudnn_sys::cudnnScaleTensor(context.as_mut_ptr(),
                                              y.0.as_ptr(),
                                              y.1.as_mut_ptr() as *mut c_void,
                                              alpha as *const S as *const c_void))
    }
    Ok(())
}

#[cfg(test)]
mod tests;
