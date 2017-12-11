use cudnn_sys;

#[derive(Clone, Copy, Debug)]
pub enum Format {
    NCHW,
    NHWC,
    NCHWVectC,
}

impl From<cudnn_sys::cudnnTensorFormat_t> for Format {
    fn from(value: cudnn_sys::cudnnTensorFormat_t) -> Format {
        match value {
            cudnn_sys::cudnnTensorFormat_t_CUDNN_TENSOR_NCHW => Format::NCHW,
            cudnn_sys::cudnnTensorFormat_t_CUDNN_TENSOR_NHWC => Format::NHWC,
            cudnn_sys::cudnnTensorFormat_t_CUDNN_TENSOR_NCHW_VECT_C => Format::NCHWVectC,
            _ => unreachable!(),
        }
    }
}

impl Into<cudnn_sys::cudnnTensorFormat_t> for Format {
    fn into(self) -> cudnn_sys::cudnnTensorFormat_t {
        match self {
            Format::NCHW => cudnn_sys::cudnnTensorFormat_t_CUDNN_TENSOR_NCHW,
            Format::NHWC => cudnn_sys::cudnnTensorFormat_t_CUDNN_TENSOR_NHWC,
            Format::NCHWVectC => cudnn_sys::cudnnTensorFormat_t_CUDNN_TENSOR_NCHW_VECT_C,
        }
    }
}
