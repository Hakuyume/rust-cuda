use cublas_sys;

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PointerMode {
    Host = cublas_sys::CUBLAS_POINTER_MODE_HOST,
    Device = cublas_sys::CUBLAS_POINTER_MODE_DEVICE,
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
