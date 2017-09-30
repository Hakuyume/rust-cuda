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

impl FwdAlgo {
    pub fn as_raw(self) -> cudnn_sys::cudnnConvolutionFwdAlgo {
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

    pub fn from_raw(fwd_algo: cudnn_sys::cudnnConvolutionFwdAlgo) -> FwdAlgo {
        match fwd_algo {
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
