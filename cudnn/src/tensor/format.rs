use cudnn_sys;
use cudnn_sys::cudnnTensorFormat::*;

#[derive(Clone, Copy, Debug)]
pub enum Format {
    NCHW,
    NHWC,
    NCHWVectC,
}

impl From<cudnn_sys::cudnnTensorFormat> for Format {
    fn from(value: cudnn_sys::cudnnTensorFormat) -> Format {
        match value {
            CUDNN_TENSOR_NCHW => Format::NCHW,
            CUDNN_TENSOR_NHWC => Format::NHWC,
            CUDNN_TENSOR_NCHW_VECT_C => Format::NCHWVectC,
        }
    }
}

impl Into<cudnn_sys::cudnnTensorFormat> for Format {
    fn into(self) -> cudnn_sys::cudnnTensorFormat {
        match self {
            Format::NCHW => CUDNN_TENSOR_NCHW,
            Format::NHWC => CUDNN_TENSOR_NHWC,
            Format::NCHWVectC => CUDNN_TENSOR_NCHW_VECT_C,
        }
    }
}
