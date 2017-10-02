use cudnn_sys;

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
            cudnn_sys::cudnnConvolutionFwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM => {
                FwdAlgo::ImplicitGemm
            }
            cudnn_sys::cudnnConvolutionFwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM =>
            FwdAlgo::ImplicitPrecompGemm,
            cudnn_sys::cudnnConvolutionFwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_GEMM => FwdAlgo::Gemm,
            cudnn_sys::cudnnConvolutionFwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT => {
                FwdAlgo::Direct
            }
            cudnn_sys::cudnnConvolutionFwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_FFT => FwdAlgo::Fft,
            cudnn_sys::cudnnConvolutionFwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING => {
                FwdAlgo::FftTiling
            }
            cudnn_sys::cudnnConvolutionFwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD => {
                FwdAlgo::Winograd
            }
            cudnn_sys::cudnnConvolutionFwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED => {
                FwdAlgo::WinogradNonfused
            }
            cudnn_sys::cudnnConvolutionFwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_COUNT => FwdAlgo::Count,
        }
    }
}

impl Into<cudnn_sys::cudnnConvolutionFwdAlgo> for FwdAlgo {
    fn into(self) -> cudnn_sys::cudnnConvolutionFwdAlgo {
        match self {
            FwdAlgo::ImplicitGemm => {
                cudnn_sys::cudnnConvolutionFwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
            }
            FwdAlgo::ImplicitPrecompGemm => {
                cudnn_sys::cudnnConvolutionFwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
            }
            FwdAlgo::Gemm => cudnn_sys::cudnnConvolutionFwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
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
