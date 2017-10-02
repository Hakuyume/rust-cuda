use cudnn_sys;

#[derive(Clone, Copy, Debug)]
pub enum Mode {
    Convolution,
    CrossCorrelation,
}

impl Into<cudnn_sys::cudnnConvolutionMode> for Mode {
    fn into(self) -> cudnn_sys::cudnnConvolutionMode {
        match self {
            Mode::Convolution => cudnn_sys::cudnnConvolutionMode::CUDNN_CONVOLUTION,
            Mode::CrossCorrelation => cudnn_sys::cudnnConvolutionMode::CUDNN_CROSS_CORRELATION,
        }
    }
}
