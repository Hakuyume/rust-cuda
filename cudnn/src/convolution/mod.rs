use cuda::memory;

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

pub fn get_2d_forward_output_dim<T: scalar::Scalar>(conv_desc: &Descriptor<T>,
                                                    input_tensor_desc: &tensor::Descriptor<T>,
                                                    filter_desc: &filter::Descriptor<T>)
                                                    -> Result<(usize, usize, usize, usize)> {
    let mut n: c_int = 0;
    let mut c: c_int = 0;
    let mut h: c_int = 0;
    let mut w: c_int = 0;
    unsafe {
        try_call!(cudnn_sys::cudnnGetConvolution2dForwardOutputDim(conv_desc.as_raw(),
                                                                   input_tensor_desc.as_raw(),
                                                                   filter_desc.as_raw(),
                                                                   &mut n,
                                                                   &mut c,
                                                                   &mut h,
                                                                   &mut w))
    }
    Ok((n as usize, c as usize, h as usize, w as usize))
}

pub fn get_forward_algorithm<T: scalar::Scalar>(context: &context::Context,
                                                x_desc: &tensor::Descriptor<T>,
                                                w_desc: &filter::Descriptor<T>,
                                                conv_desc: &Descriptor<T>,
                                                y_desc: &tensor::Descriptor<T>,
                                                requested_algo_count: usize)
                                                -> Result<Vec<FwdAlgoPerf>> {
    let mut returned_algo_count = 0;
    let mut perf_results: Vec<cudnn_sys::cudnnConvolutionFwdAlgoPerf> =
        Vec::with_capacity(requested_algo_count);
    unsafe {
        try_call!(cudnn_sys::cudnnFindConvolutionForwardAlgorithm(context.as_raw(),
                                                                  x_desc.as_raw(),
                                                                  w_desc.as_raw(),
                                                                  conv_desc.as_raw(),
                                                                  y_desc.as_raw(),
                                                                  requested_algo_count as c_int,
                                                                  &mut returned_algo_count,
                                                                  perf_results.as_mut_ptr()));
        perf_results.set_len(returned_algo_count as usize);
    }
    Ok(perf_results
           .into_iter()
           .map(|fwd_algo_perf| FwdAlgoPerf::from_raw(fwd_algo_perf))
           .collect())
}

pub fn get_forward_workspace_size<T: scalar::Scalar>(context: &context::Context,
                                                     x_desc: &tensor::Descriptor<T>,
                                                     w_desc: &filter::Descriptor<T>,
                                                     conv_desc: &Descriptor<T>,
                                                     y_desc: &tensor::Descriptor<T>,
                                                     algo: FwdAlgo)
                                                     -> Result<usize> {
    let mut size: size_t = 0;
    unsafe {
        try_call!(cudnn_sys::cudnnGetConvolutionForwardWorkspaceSize(context.as_raw(),
                                                                     x_desc.as_raw(),
                                                                     w_desc.as_raw(),
                                                                     conv_desc.as_raw(),
                                                                     y_desc.as_raw(),
                                                                     algo.as_raw(),
                                                                     &mut size))
    }
    Ok(size as usize)
}

pub fn forward<'a, T>(context: &context::Context,
                      alpha: T,
                      x: tensor::Tensor<'a, T>,
                      w: filter::Filter<'a, T>,
                      conv_desc: &Descriptor<T>,
                      algo: FwdAlgo,
                      workspace: &mut memory::Slice<u8>,
                      beta: T,
                      y: tensor::TensorMut<'a, T>)
                      -> Result<()>
    where T: scalar::Float
{
    let params: &[T::Scale] = &[alpha.into(), beta.into()];
    unsafe {
        try_call!(cudnn_sys::cudnnConvolutionForward(context.as_raw(),
                                                     &params[0] as *const T::Scale as
                                                     *const c_void,
                                                     x.desc.as_raw(),
                                                     x.mem.as_ptr() as *const c_void,
                                                     w.desc.as_raw(),
                                                     w.mem.as_ptr() as *const c_void,
                                                     conv_desc.as_raw(),
                                                     algo.as_raw(),
                                                     workspace.as_mut_ptr() as *mut c_void,
                                                     workspace.len(),
                                                     &params[1] as *const T::Scale as
                                                     *const c_void,
                                                     y.desc.as_raw(),
                                                     y.mem.as_mut_ptr() as *mut c_void))
    }
    Ok(())
}
