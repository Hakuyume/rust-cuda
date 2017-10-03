use cuda_sys;
use cuda_sys::{c_void, size_t};

use Result;
use stream;

pub type Dim3 = cuda_sys::dim3;

pub unsafe fn launch_kernel<T: ?Sized>(func: *const T,
                                       grid_dim: Dim3,
                                       block_dim: Dim3,
                                       args: &mut [*mut c_void],
                                       shared_mem: usize,
                                       stream: &mut stream::Stream)
                                       -> Result<()> {
    try_call!(cuda_sys::cudaLaunchKernel(func as *const c_void,
                                         grid_dim.into(),
                                         block_dim.into(),
                                         args.as_mut_ptr(),
                                         shared_mem as size_t,
                                         stream.as_mut_ptr()));
    Ok(())
}
