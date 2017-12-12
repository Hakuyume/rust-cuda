use std::error;
use std::fmt;
use std::result;

use cublas_sys;

#[derive(Clone, Copy, Debug)]
pub enum Error {
    NotInitialized,
    AllocFailed,
    InvalidValue,
    ArchMismatch,
    MappingError,
    ExecutionFailed,
    InternalError,
    NotSupported,
    LicenseError,
}

pub trait TryFrom<T>: Sized {
    type Error;
    fn try_from(T) -> result::Result<Self, Self::Error>;
}

impl TryFrom<cublas_sys::cublasStatus_t> for Error {
    type Error = ();
    fn try_from(value: cublas_sys::cublasStatus_t) -> result::Result<Error, ()> {
        match value {
            cublas_sys::CUBLAS_STATUS_SUCCESS => Err(()),
            cublas_sys::CUBLAS_STATUS_NOT_INITIALIZED => Ok(Error::NotInitialized),
            cublas_sys::CUBLAS_STATUS_ALLOC_FAILED => Ok(Error::AllocFailed),
            cublas_sys::CUBLAS_STATUS_INVALID_VALUE => Ok(Error::InvalidValue),
            cublas_sys::CUBLAS_STATUS_ARCH_MISMATCH => Ok(Error::ArchMismatch),
            cublas_sys::CUBLAS_STATUS_MAPPING_ERROR => Ok(Error::MappingError),
            cublas_sys::CUBLAS_STATUS_EXECUTION_FAILED => Ok(Error::ExecutionFailed),
            cublas_sys::CUBLAS_STATUS_INTERNAL_ERROR => Ok(Error::InternalError),
            cublas_sys::CUBLAS_STATUS_NOT_SUPPORTED => Ok(Error::NotSupported),
            cublas_sys::CUBLAS_STATUS_LICENSE_ERROR => Ok(Error::LicenseError),
            _ => unreachable!(),
        }
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        match self {
            &Error::NotInitialized => "The cuBLAS library was not initialized.",
            &Error::AllocFailed => "Resource allocation failed inside the cuBLAS library.",
            &Error::InvalidValue => "An unsupported value or parameter was passed to the function (a negative vector size, for example).",
            &Error::ArchMismatch => "The function requires a feature absent from the device architecture; usually caused by the lack of support for double precision.",
            &Error::MappingError => "An access to GPU memory space failed, which is usually caused by a failure to bind a texture.",
            &Error::ExecutionFailed => "The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.",
            &Error::InternalError => "An internal cuBLAS operation failed.",
            &Error::NotSupported => "The functionnality requested is not supported.",
            &Error::LicenseError => "The functionnality requested requires some license and an error was detected when trying to check the current licensing.",
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

macro_rules! try_call {
    ($call:expr) => {{
        use $crate::error::TryFrom;
        try!(match $crate::Error::try_from($call) {
            Ok(err) => Err(err),
            Err(_) => Ok(()),
        })
    }};
}
