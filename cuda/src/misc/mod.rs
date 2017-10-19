use std::ptr;

use cuda_sys;
use cuda_sys::{c_void, size_t};

use Result;
use stream;

mod dim3;
pub use self::dim3::Dim3;

pub trait Arg {}
impl<T> Arg for T {}

pub unsafe fn launch_kernel(func: *const c_void,
                            grid_dim: Dim3,
                            block_dim: Dim3,
                            args: &[&Arg],
                            shared_mem: usize,
                            stream: Option<&mut stream::Stream>)
                            -> Result<()> {
    let args: Vec<_> = args.iter()
        .map(|arg| *arg as *const Arg as *const c_void)
        .collect();
    try_call!(cuda_sys::cudaLaunchKernel(func,
                                         grid_dim.into(),
                                         block_dim.into(),
                                         args.as_ptr(),
                                         shared_mem as size_t,
                                         match stream {
                                             Some(stream) => stream.as_mut_ptr(),
                                             None => ptr::null_mut(),
                                         }));
    Ok(())
}
