use std::any;
use std::os::raw::c_void;
use std::ptr;

use cuda_sys;

use Result;
use stream;

mod dim3;
pub use self::dim3::Dim3;

pub unsafe fn launch_kernel(func: *const c_void,
                            grid_dim: Dim3,
                            block_dim: Dim3,
                            args: &mut [&mut any::Any],
                            shared_mem: usize,
                            stream: Option<&mut stream::Stream>)
                            -> Result<()> {
    let mut args: Vec<_> = args.iter_mut()
        .map(|arg| *arg as *mut any::Any as *mut c_void)
        .collect();
    try_call!(cuda_sys::cudaLaunchKernel(func,
                                         grid_dim.into(),
                                         block_dim.into(),
                                         args.as_mut_ptr(),
                                         shared_mem,
                                         match stream {
                                             Some(stream) => stream.as_mut_ptr(),
                                             None => ptr::null_mut(),
                                         }));
    Ok(())
}
