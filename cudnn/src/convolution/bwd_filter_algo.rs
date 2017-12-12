use cudnn_sys;

#[derive(Clone, Copy, Debug)]
pub enum BwdFilterAlgo {
    _0,
    _1,
    Fft,
    _3,
    Winograd,
    WinogradNonfused,
    FftTitiling,
    Count,
}

impl From<cudnn_sys::cudnnConvolutionBwdFilterAlgo_t> for BwdFilterAlgo {
    fn from(value: cudnn_sys::cudnnConvolutionBwdFilterAlgo_t) -> BwdFilterAlgo {
        match value {
            cudnn_sys::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 => BwdFilterAlgo::_0,
            cudnn_sys::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 => BwdFilterAlgo::_1,
            cudnn_sys::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT => BwdFilterAlgo::Fft,
            cudnn_sys::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 => BwdFilterAlgo::_3,
            cudnn_sys::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD => BwdFilterAlgo::Winograd,
            cudnn_sys::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED => {
                BwdFilterAlgo::WinogradNonfused
            }
            cudnn_sys::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING => BwdFilterAlgo::FftTitiling,
            cudnn_sys::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT => BwdFilterAlgo::Count,
            _ => unreachable!(),
        }
    }
}

impl Into<cudnn_sys::cudnnConvolutionBwdFilterAlgo_t> for BwdFilterAlgo {
    fn into(self) -> cudnn_sys::cudnnConvolutionBwdFilterAlgo_t {
        match self {
            BwdFilterAlgo::_0 => cudnn_sys::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
            BwdFilterAlgo::_1 => cudnn_sys::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
            BwdFilterAlgo::Fft => cudnn_sys::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
            BwdFilterAlgo::_3 => cudnn_sys::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
            BwdFilterAlgo::Winograd => cudnn_sys::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD,
            BwdFilterAlgo::WinogradNonfused => {
                cudnn_sys::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED
            }
            BwdFilterAlgo::FftTitiling => cudnn_sys::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,
            BwdFilterAlgo::Count => cudnn_sys::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT,
        }
    }
}
