use cudnn_sys;

#[derive(Clone, Copy, Debug)]
pub enum Format {
    NCHW,
    NHWC,
}

impl Into<cudnn_sys::cudnnTensorFormat> for Format {
    fn into(self) -> cudnn_sys::cudnnTensorFormat {
        match self {
            Format::NCHW => cudnn_sys::cudnnTensorFormat::CUDNN_TENSOR_NCHW,
            Format::NHWC => cudnn_sys::cudnnTensorFormat::CUDNN_TENSOR_NHWC,
        }
    }
}
