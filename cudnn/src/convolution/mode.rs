use cudnn_sys;

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum Mode {
    Convolution = cudnn_sys::CUDNN_CONVOLUTION,
    CrossCorrelation = cudnn_sys::CUDNN_CROSS_CORRELATION,
}
