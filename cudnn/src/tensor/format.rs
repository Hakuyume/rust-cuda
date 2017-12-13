use cudnn_sys;

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum Format {
    NCHW = cudnn_sys::CUDNN_TENSOR_NCHW,
    NHWC = cudnn_sys::CUDNN_TENSOR_NHWC,
    NCHWVectC = cudnn_sys::CUDNN_TENSOR_NCHW_VECT_C,
}

impl From<cudnn_sys::cudnnTensorFormat_t> for Format {
    fn from(value: cudnn_sys::cudnnTensorFormat_t) -> Format {
        match value {
            cudnn_sys::CUDNN_TENSOR_NCHW => Format::NCHW,
            cudnn_sys::CUDNN_TENSOR_NHWC => Format::NHWC,
            cudnn_sys::CUDNN_TENSOR_NCHW_VECT_C => Format::NCHWVectC,
            _ => unreachable!(),
        }
    }
}
