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

mod fwd_preference;
pub use self::fwd_preference::FwdPreference;

mod fwd_algo;
pub use self::fwd_algo::FwdAlgo;

mod fwd_algo_perf;
pub use self::fwd_algo_perf::FwdAlgoPerf;

mod bwd_filter_preference;
pub use self::bwd_filter_preference::BwdFilterPreference;

mod bwd_filter_algo;
pub use self::bwd_filter_algo::BwdFilterAlgo;

use misc::MemoryDescriptor;

pub fn get_2d_forward_output_dim<T>(conv_desc: &Descriptor<T>,
                                    input_tensor_desc: &tensor::Descriptor<T>,
                                    filter_desc: &filter::Descriptor<T>)
                                    -> Result<(usize, usize, usize, usize)>
    where T: scalar::Scalar
{
    let mut n = 0;
    let mut c = 0;
    let mut h = 0;
    let mut w = 0;
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
    let mut perf_results = Vec::with_capacity(requested_algo_count);
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
    let mut size_in_bytes = 0;
    unsafe {
        try_call!(cudnn_sys::cudnnGetConvolutionForwardWorkspaceSize(context.as_mut_ptr(),
                                                                     x_desc.as_ptr(),
                                                                     w_desc.as_ptr(),
                                                                     conv_desc.as_ptr(),
                                                                     y_desc.as_ptr(),
                                                                     algo.into(),
                                                                     &mut size_in_bytes))
    }
    Ok(size_in_bytes as usize)
}

pub fn forward<T, S, X, W, R, Y>(context: &mut context::Context,
                                 alpha: S,
                                 x: (&tensor::Descriptor<T>, &X),
                                 w: (&filter::Descriptor<T>, &W),
                                 conv_desc: &Descriptor<T>,
                                 algo: FwdAlgo,
                                 workspace: &mut R,
                                 beta: S,
                                 y: (&tensor::Descriptor<T>, &mut Y))
                                 -> Result<()>
    where T: scalar::Scalar + scalar::Scale<Scale = S>,
          X: Repr<T>,
          W: Repr<T>,
          R: ReprMut<u8>,
          Y: ReprMut<T>
{
    x.0.check_memory(x.1)?;
    w.0.check_memory(w.1)?;
    y.0.check_memory(y.1)?;
    unsafe {
        try_call!(cudnn_sys::cudnnConvolutionForward(context.as_mut_ptr(),
                                                     &alpha as *const S as *const c_void,
                                                     x.0.as_ptr(),
                                                     x.1.as_ptr() as *const c_void,
                                                     w.0.as_ptr(),
                                                     w.1.as_ptr() as *const c_void,
                                                     conv_desc.as_ptr(),
                                                     algo.into(),
                                                     workspace.as_mut_ptr() as *mut c_void,
                                                     workspace.len() as size_t,
                                                     &beta as *const S as *const c_void,
                                                     y.0.as_ptr(),
                                                     y.1.as_mut_ptr() as *mut c_void))
    }
    Ok(())
}

pub fn get_backward_filter_algorithm<T>(context: &mut context::Context,
                                        x_desc: &tensor::Descriptor<T>,
                                        dy_desc: &tensor::Descriptor<T>,
                                        conv_desc: &Descriptor<T>,
                                        dw_desc: &filter::Descriptor<T>,
                                        preference: BwdFilterPreference)
                                        -> Result<BwdFilterAlgo>
    where T: scalar::Scalar
{
    let (preference, memory_limit_in_bytes) = preference.into();
    let mut algo = cudnn_sys::cudnnConvolutionBwdFilterAlgo::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
    unsafe {
        try_call!(cudnn_sys::cudnnGetConvolutionBackwardFilterAlgorithm(context.as_mut_ptr(),
                                                                        x_desc.as_ptr(),
                                                                        dy_desc.as_ptr(),
                                                                        conv_desc.as_ptr(),
                                                                        dw_desc.as_ptr(),
                                                                        preference,
                                                                        memory_limit_in_bytes
                                                                            .unwrap_or(0) as
                                                                        size_t,
                                                                        &mut algo))
    }
    Ok(algo.into())
}

pub fn get_backward_filter_workspace_size<T>(context: &mut context::Context,
                                             x_desc: &tensor::Descriptor<T>,
                                             dy_desc: &tensor::Descriptor<T>,
                                             conv_desc: &Descriptor<T>,
                                             dw_desc: &filter::Descriptor<T>,
                                             algo: BwdFilterAlgo)
                                             -> Result<usize>
    where T: scalar::Scalar
{
    let mut size_in_bytes = 0;
    unsafe {
        try_call!(cudnn_sys::cudnnGetConvolutionBackwardFilterWorkspaceSize(context.as_mut_ptr(),
                                                                            x_desc.as_ptr(),
                                                                            dy_desc.as_ptr(),
                                                                            conv_desc.as_ptr(),
                                                                            dw_desc.as_ptr(),
                                                                            algo.into(),
                                                                            &mut size_in_bytes))
    }
    Ok(size_in_bytes as usize)
}

pub fn backward_filter<T, S, X, Dy, R, Dw>(context: &mut context::Context,
                                           alpha: S,
                                           x: (&tensor::Descriptor<T>, &X),
                                           dy: (&tensor::Descriptor<T>, &Dy),
                                           conv_desc: &Descriptor<T>,
                                           algo: BwdFilterAlgo,
                                           workspace: &mut R,
                                           beta: S,
                                           dw: (&filter::Descriptor<T>, &mut Dw))
                                           -> Result<()>
    where T: scalar::Scalar + scalar::Scale<Scale = S>,
          X: Repr<T>,
          Dy: Repr<T>,
          R: ReprMut<u8>,
          Dw: ReprMut<T>
{
    x.0.check_memory(x.1)?;
    dy.0.check_memory(dy.1)?;
    dw.0.check_memory(dw.1)?;
    unsafe {
        try_call!(cudnn_sys::cudnnConvolutionBackwardFilter(context.as_mut_ptr(),
                                                            &alpha as *const S as *const c_void,
                                                            x.0.as_ptr(),
                                                            x.1.as_ptr() as *const c_void,
                                                            dy.0.as_ptr(),
                                                            dy.1.as_ptr() as *const c_void,
                                                            conv_desc.as_ptr(),
                                                            algo.into(),
                                                            workspace.as_mut_ptr() as *mut c_void,
                                                            workspace.len() as size_t,
                                                            &beta as *const S as *const c_void,
                                                            dw.0.as_ptr(),
                                                            dw.1.as_mut_ptr() as *mut c_void))
    }
    Ok(())
}
