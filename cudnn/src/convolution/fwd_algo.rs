use cudnn_sys;
use cudnn_sys::cudnnConvolutionFwdAlgo::*;

#[derive(Clone, Copy, Debug)]
pub enum FwdAlgo {
    ImplicitGemm,
    ImplicitPrecompGemm,
    Gemm,
    Direct,
    Fft,
    FftTiling,
    Winograd,
    WinogradNonfused,
    Count,
}

impl From<cudnn_sys::cudnnConvolutionFwdAlgo> for FwdAlgo {
    fn from(value: cudnn_sys::cudnnConvolutionFwdAlgo) -> FwdAlgo {
        match value {
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM => FwdAlgo::ImplicitGemm,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM => FwdAlgo::ImplicitPrecompGemm,
            CUDNN_CONVOLUTION_FWD_ALGO_GEMM => FwdAlgo::Gemm,
            CUDNN_CONVOLUTION_FWD_ALGO_DIRECT => FwdAlgo::Direct,
            CUDNN_CONVOLUTION_FWD_ALGO_FFT => FwdAlgo::Fft,
            CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING => FwdAlgo::FftTiling,
            CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD => FwdAlgo::Winograd,
            CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED => FwdAlgo::WinogradNonfused,
            CUDNN_CONVOLUTION_FWD_ALGO_COUNT => FwdAlgo::Count,
        }
    }
}

impl Into<cudnn_sys::cudnnConvolutionFwdAlgo> for FwdAlgo {
    fn into(self) -> cudnn_sys::cudnnConvolutionFwdAlgo {
        match self {
            FwdAlgo::ImplicitGemm => CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            FwdAlgo::ImplicitPrecompGemm => CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
            FwdAlgo::Gemm => CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
            FwdAlgo::Direct => CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
            FwdAlgo::Fft => CUDNN_CONVOLUTION_FWD_ALGO_FFT,
            FwdAlgo::FftTiling => CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
            FwdAlgo::Winograd => CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
            FwdAlgo::WinogradNonfused => CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
            FwdAlgo::Count => CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
        }
    }
}
