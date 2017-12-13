use cudnn_sys;

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum FwdAlgo {
    ImplicitGemm = cudnn_sys::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
    ImplicitPrecompGemm = cudnn_sys::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
    Gemm = cudnn_sys::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
    Direct = cudnn_sys::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
    Fft = cudnn_sys::CUDNN_CONVOLUTION_FWD_ALGO_FFT,
    FftTiling = cudnn_sys::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
    Winograd = cudnn_sys::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
    WinogradNonfused = cudnn_sys::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
    Count = cudnn_sys::CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
}

impl From<cudnn_sys::cudnnConvolutionFwdAlgo_t> for FwdAlgo {
    fn from(value: cudnn_sys::cudnnConvolutionFwdAlgo_t) -> FwdAlgo {
        match value {
            cudnn_sys::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM => FwdAlgo::ImplicitGemm,
            cudnn_sys::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM => {
                FwdAlgo::ImplicitPrecompGemm
            }
            cudnn_sys::CUDNN_CONVOLUTION_FWD_ALGO_GEMM => FwdAlgo::Gemm,
            cudnn_sys::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT => FwdAlgo::Direct,
            cudnn_sys::CUDNN_CONVOLUTION_FWD_ALGO_FFT => FwdAlgo::Fft,
            cudnn_sys::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING => FwdAlgo::FftTiling,
            cudnn_sys::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD => FwdAlgo::Winograd,
            cudnn_sys::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED => FwdAlgo::WinogradNonfused,
            cudnn_sys::CUDNN_CONVOLUTION_FWD_ALGO_COUNT => FwdAlgo::Count,
            _ => unreachable!(),
        }
    }
}
