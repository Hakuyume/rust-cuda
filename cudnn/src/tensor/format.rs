use cudnn_sys;
use cudnn_sys::cudnnTensorFormat::*;

#[derive(Clone, Copy, Debug)]
pub enum Format {
    NCHW,
    NHWC,
}

impl Into<cudnn_sys::cudnnTensorFormat> for Format {
    fn into(self) -> cudnn_sys::cudnnTensorFormat {
        match self {
            Format::NCHW => CUDNN_TENSOR_NCHW,
            Format::NHWC => CUDNN_TENSOR_NHWC,
        }
    }
}
