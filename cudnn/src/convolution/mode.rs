use cudnn_sys;

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
