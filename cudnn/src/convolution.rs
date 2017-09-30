use std::marker;
use std::ptr;

use cudnn_sys;
use cudnn_sys::{c_int, size_t};

use scalar;
use Result;
use context;
use tensor;
use filter;

pub enum Mode {
    Convolution,
    CrossCorrelation,
}

impl Mode {
    pub fn as_raw(self) -> cudnn_sys::cudnnConvolutionMode {
        match self {
            Mode::Convolution => cudnn_sys::cudnnConvolutionMode::CUDNN_CONVOLUTION,
            Mode::CrossCorrelation => cudnn_sys::cudnnConvolutionMode::CUDNN_CROSS_CORRELATION,
        }
    }
}

pub struct ConvolutionDescriptor<T: scalar::Scalar> {
    desc: cudnn_sys::cudnnConvolutionDescriptor,
    _dummy: marker::PhantomData<T>,
}

impl<T: scalar::Scalar> ConvolutionDescriptor<T> {
    fn new() -> Result<ConvolutionDescriptor<T>> {
        let mut desc = ptr::null_mut();
        unsafe { try_call!(cudnn_sys::cudnnCreateConvolutionDescriptor(&mut desc)) }
        Ok(ConvolutionDescriptor {
               desc,
               _dummy: marker::PhantomData::default(),
           })
    }

    pub fn new_2d(pad_h: usize,
                  pad_w: usize,
                  u: usize,
                  v: usize,
                  dilation_h: usize,
                  dilation_w: usize,
                  mode: Mode)
                  -> Result<ConvolutionDescriptor<T>> {
        let desc = try!(ConvolutionDescriptor::new());
        unsafe {
            try_call!(cudnn_sys::cudnnSetConvolution2dDescriptor(desc.as_raw(),
                                                                 pad_h as c_int,
                                                                 pad_w as c_int,
                                                                 u as c_int,
                                                                 v as c_int,
                                                                 dilation_h as c_int,
                                                                 dilation_w as c_int,
                                                                 mode.as_raw(),
                                                                 T::DATA_TYPE))
        };

        Ok(desc)
    }

    pub fn as_raw(&self) -> cudnn_sys::cudnnConvolutionDescriptor {
        self.desc
    }
}

impl<T: scalar::Scalar> Drop for ConvolutionDescriptor<T> {
    fn drop(&mut self) {
        unsafe { cudnn_sys::cudnnDestroyConvolutionDescriptor(self.desc) };
    }
}

#[repr(C)]
pub enum FwdAlgo {
    ImplicitGemm,
    ImplicitPrecompGemm,
    AlgoGemm,
    Direct,
    Fft,
    FftTiling,
    Winograd,
    WinogradNonfused,
    Count,
}

impl FwdAlgo {
    pub fn as_raw(self) -> cudnn_sys::cudnnConvolutionFwdAlgo {
        match self {
            FwdAlgo::ImplicitGemm => {
                cudnn_sys::cudnnConvolutionFwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
            }
            FwdAlgo::ImplicitPrecompGemm => {
                cudnn_sys::cudnnConvolutionFwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
            }
            FwdAlgo::AlgoGemm => {
                cudnn_sys::cudnnConvolutionFwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_GEMM
            }
            FwdAlgo::Direct => {
                cudnn_sys::cudnnConvolutionFwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT
            }
            FwdAlgo::Fft => cudnn_sys::cudnnConvolutionFwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_FFT,
            FwdAlgo::FftTiling => {
                cudnn_sys::cudnnConvolutionFwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING
            }
            FwdAlgo::Winograd => {
                cudnn_sys::cudnnConvolutionFwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD
            }
            FwdAlgo::WinogradNonfused => {
                cudnn_sys::cudnnConvolutionFwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
            }
            FwdAlgo::Count => cudnn_sys::cudnnConvolutionFwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
        }
    }
}

pub fn get_forward_workspace_size<T: scalar::Scalar>(context: &context::Context,
                                                     x_desc: &tensor::TensorDescriptor<T>,
                                                     w_desc: &filter::FilterDescriptor<T>,
                                                     conv_desc: &ConvolutionDescriptor<T>,
                                                     y_desc: &tensor::TensorDescriptor<T>,
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
