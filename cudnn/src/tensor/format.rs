use cudnn_sys;

pub enum Format {
    NCHW,
    NHWC,
}

impl Format {
    pub fn as_raw(self) -> cudnn_sys::cudnnTensorFormat {
        match self {
            Format::NCHW => cudnn_sys::cudnnTensorFormat::CUDNN_TENSOR_NCHW,
            Format::NHWC => cudnn_sys::cudnnTensorFormat::CUDNN_TENSOR_NHWC,
        }
    }
}
