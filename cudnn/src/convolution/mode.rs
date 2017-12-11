use cudnn_sys;

#[derive(Clone, Copy, Debug)]
pub enum Mode {
    Convolution,
    CrossCorrelation,
}

impl Into<cudnn_sys::cudnnConvolutionMode_t> for Mode {
    fn into(self) -> cudnn_sys::cudnnConvolutionMode_t {
        match self {
            Mode::Convolution => cudnn_sys::cudnnConvolutionMode_t_CUDNN_CONVOLUTION,
            Mode::CrossCorrelation => cudnn_sys::cudnnConvolutionMode_t_CUDNN_CROSS_CORRELATION,
        }
    }
}
