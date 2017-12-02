use cudnn_sys;
use cudnn_sys::cudnnConvolutionBwdFilterAlgo::*;

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

impl From<cudnn_sys::cudnnConvolutionBwdFilterAlgo> for BwdFilterAlgo {
    fn from(value: cudnn_sys::cudnnConvolutionBwdFilterAlgo) -> BwdFilterAlgo {
        match value {
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 => BwdFilterAlgo::_0,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 => BwdFilterAlgo::_1,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT => BwdFilterAlgo::Fft,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 => BwdFilterAlgo::_3,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD => BwdFilterAlgo::Winograd,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED => BwdFilterAlgo::WinogradNonfused,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING => BwdFilterAlgo::FftTitiling,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT => BwdFilterAlgo::Count,
        }
    }
}

impl Into<cudnn_sys::cudnnConvolutionBwdFilterAlgo> for BwdFilterAlgo {
    fn into(self) -> cudnn_sys::cudnnConvolutionBwdFilterAlgo {
        match self {
            BwdFilterAlgo::_0 => CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
            BwdFilterAlgo::_1 => CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
            BwdFilterAlgo::Fft => CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
            BwdFilterAlgo::_3 => CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
            BwdFilterAlgo::Winograd => CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD,
            BwdFilterAlgo::WinogradNonfused => CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
            BwdFilterAlgo::FftTitiling => CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,
            BwdFilterAlgo::Count => CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT,
        }
    }
}
