use cuda::memory::{Repr, ReprMut};

use cudnn_sys;
use cudnn_sys::{c_int, c_void, size_t};

use scalar;
use Result;
use context;
use tensor;
use filter;

mod mode;
pub use self::mode::Mode;

mod descriptor;
pub use self::descriptor::Descriptor;

mod fwd_algo;
pub use self::fwd_algo::FwdAlgo;

mod fwd_algo_perf;
pub use self::fwd_algo_perf::FwdAlgoPerf;

mod fwd_preference;
pub use self::fwd_preference::FwdPreference;

pub fn get_2d_forward_output_dim<T>(conv_desc: &Descriptor<T>,
                                    input_tensor_desc: &tensor::Descriptor<T>,
                                    filter_desc: &filter::Descriptor<T>)
                                    -> Result<(usize, usize, usize, usize)>
    where T: scalar::Scalar
{
    let mut n: c_int = 0;
    let mut c: c_int = 0;
    let mut h: c_int = 0;
    let mut w: c_int = 0;
    unsafe {
        try_call!(cudnn_sys::cudnnGetConvolution2dForwardOutputDim(conv_desc.as_ptr(),
                                                                   input_tensor_desc.as_ptr(),
                                                                   filter_desc.as_ptr(),
                                                                   &mut n,
                                                                   &mut c,
                                                                   &mut h,
                                                                   &mut w))
    }
    Ok((n as usize, c as usize, h as usize, w as usize))
}

pub fn find_forward_algorithm<T>(context: &mut context::Context,
                                 x_desc: &tensor::Descriptor<T>,
                                 w_desc: &filter::Descriptor<T>,
                                 conv_desc: &Descriptor<T>,
                                 y_desc: &tensor::Descriptor<T>,
                                 requested_algo_count: usize)
                                 -> Result<Vec<FwdAlgoPerf>>
    where T: scalar::Scalar
{
    let mut returned_algo_count = 0;
    let mut perf_results: Vec<cudnn_sys::cudnnConvolutionFwdAlgoPerf> =
        Vec::with_capacity(requested_algo_count);
    unsafe {
        try_call!(cudnn_sys::cudnnFindConvolutionForwardAlgorithm(context.as_mut_ptr(),
                                                                  x_desc.as_ptr(),
                                                                  w_desc.as_ptr(),
                                                                  conv_desc.as_ptr(),
                                                                  y_desc.as_ptr(),
                                                                  requested_algo_count as c_int,
                                                                  &mut returned_algo_count,
                                                                  perf_results.as_mut_ptr()));
        perf_results.set_len(returned_algo_count as usize);
    }
    Ok(perf_results
           .into_iter()
           .map(|fwd_algo_perf| fwd_algo_perf.into())
           .collect())
}

pub fn get_forward_algorithm<T>(context: &mut context::Context,
                                x_desc: &tensor::Descriptor<T>,
                                w_desc: &filter::Descriptor<T>,
                                conv_desc: &Descriptor<T>,
                                y_desc: &tensor::Descriptor<T>,
                                preference: FwdPreference)
                                -> Result<FwdAlgo>
    where T: scalar::Scalar
{
    let (preference, memory_limit_in_bytes) = preference.into();
    let mut algo = cudnn_sys::cudnnConvolutionFwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    unsafe {
        try_call!(cudnn_sys::cudnnGetConvolutionForwardAlgorithm(context.as_mut_ptr(),
                                                                 x_desc.as_ptr(),
                                                                 w_desc.as_ptr(),
                                                                 conv_desc.as_ptr(),
                                                                 y_desc.as_ptr(),
                                                                 preference,
                                                                 memory_limit_in_bytes
                                                                     .unwrap_or(0) as
                                                                 size_t,
                                                                 &mut algo));
    }
    Ok(algo.into())
}

pub fn get_forward_workspace_size<T>(context: &mut context::Context,
                                     x_desc: &tensor::Descriptor<T>,
                                     w_desc: &filter::Descriptor<T>,
                                     conv_desc: &Descriptor<T>,
                                     y_desc: &tensor::Descriptor<T>,
                                     algo: FwdAlgo)
                                     -> Result<usize>
    where T: scalar::Scalar
{
    let mut size: size_t = 0;
    unsafe {
        try_call!(cudnn_sys::cudnnGetConvolutionForwardWorkspaceSize(context.as_mut_ptr(),
                                                                     x_desc.as_ptr(),
                                                                     w_desc.as_ptr(),
                                                                     conv_desc.as_ptr(),
                                                                     y_desc.as_ptr(),
                                                                     algo.into(),
                                                                     &mut size))
    }
    Ok(size as usize)
}

pub fn forward<'a, T, S, R>(context: &mut context::Context,
                            alpha: S,
                            x: tensor::Tensor<'a, T>,
                            w: filter::Filter<'a, T>,
                            conv_desc: &Descriptor<T>,
                            algo: FwdAlgo,
                            workspace: &mut R,
                            beta: S,
                            mut y: tensor::TensorMut<'a, T>)
                            -> Result<()>
    where T: scalar::Scalar + scalar::Scale<Scale = S>,
          R: ReprMut<u8>
{
    unsafe {
        try_call!(cudnn_sys::cudnnConvolutionForward(context.as_mut_ptr(),
                                                     &alpha as *const T::Scale as *const c_void,
                                                     x.desc().as_ptr(),
                                                     x.mem().as_ptr() as *const c_void,
                                                     w.desc().as_ptr(),
                                                     w.mem().as_ptr() as *const c_void,
                                                     conv_desc.as_ptr(),
                                                     algo.into(),
                                                     workspace.as_mut_ptr() as *mut c_void,
                                                     workspace.len(),
                                                     &beta as *const T::Scale as *const c_void,
                                                     y.desc().as_ptr(),
                                                     y.mem_mut().as_mut_ptr() as *mut c_void))
    }
    Ok(())
}
