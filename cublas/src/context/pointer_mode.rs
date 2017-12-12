use cublas_sys;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PointerMode {
    Host,
    Device,
}

impl From<cublas_sys::cublasPointerMode_t> for PointerMode {
    fn from(value: cublas_sys::cublasPointerMode_t) -> PointerMode {
        match value {
            cublas_sys::CUBLAS_POINTER_MODE_HOST => PointerMode::Host,
            cublas_sys::CUBLAS_POINTER_MODE_DEVICE => PointerMode::Device,
            _ => unreachable!(),
        }
    }
}

impl Into<cublas_sys::cublasPointerMode_t> for PointerMode {
    fn into(self) -> cublas_sys::cublasPointerMode_t {
        match self {
            PointerMode::Host => cublas_sys::CUBLAS_POINTER_MODE_HOST,
            PointerMode::Device => cublas_sys::CUBLAS_POINTER_MODE_DEVICE,
        }
    }
}
