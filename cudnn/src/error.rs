use std::error;
use std::ffi;
use std::fmt;
use std::result;

use cudnn_sys;

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum Error {
    NotInitialized = cudnn_sys::CUDNN_STATUS_NOT_INITIALIZED,
    AllocFailed = cudnn_sys::CUDNN_STATUS_ALLOC_FAILED,
    BadParam = cudnn_sys::CUDNN_STATUS_BAD_PARAM,
    InternalError = cudnn_sys::CUDNN_STATUS_INTERNAL_ERROR,
    InvalidValue = cudnn_sys::CUDNN_STATUS_INVALID_VALUE,
    ArchMismatch = cudnn_sys::CUDNN_STATUS_ARCH_MISMATCH,
    MappingError = cudnn_sys::CUDNN_STATUS_MAPPING_ERROR,
    ExecutionFailed = cudnn_sys::CUDNN_STATUS_EXECUTION_FAILED,
    NotSupported = cudnn_sys::CUDNN_STATUS_NOT_SUPPORTED,
    LicenseError = cudnn_sys::CUDNN_STATUS_LICENSE_ERROR,
    RuntimePrerequisiteMissing = cudnn_sys::CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING,
}

pub trait TryFrom<T>: Sized {
    type Error;
    fn try_from(T) -> result::Result<Self, Self::Error>;
}

impl TryFrom<cudnn_sys::cudnnStatus_t> for Error {
    type Error = ();
    fn try_from(value: cudnn_sys::cudnnStatus_t) -> result::Result<Error, ()> {
        match value {
            cudnn_sys::CUDNN_STATUS_SUCCESS => Err(()),
            cudnn_sys::CUDNN_STATUS_NOT_INITIALIZED => Ok(Error::NotInitialized),
            cudnn_sys::CUDNN_STATUS_ALLOC_FAILED => Ok(Error::AllocFailed),
            cudnn_sys::CUDNN_STATUS_BAD_PARAM => Ok(Error::BadParam),
            cudnn_sys::CUDNN_STATUS_INTERNAL_ERROR => Ok(Error::InternalError),
            cudnn_sys::CUDNN_STATUS_INVALID_VALUE => Ok(Error::InvalidValue),
            cudnn_sys::CUDNN_STATUS_ARCH_MISMATCH => Ok(Error::ArchMismatch),
            cudnn_sys::CUDNN_STATUS_MAPPING_ERROR => Ok(Error::MappingError),
            cudnn_sys::CUDNN_STATUS_EXECUTION_FAILED => Ok(Error::ExecutionFailed),
            cudnn_sys::CUDNN_STATUS_NOT_SUPPORTED => Ok(Error::NotSupported),
            cudnn_sys::CUDNN_STATUS_LICENSE_ERROR => Ok(Error::LicenseError),
            cudnn_sys::CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING => {
                Ok(Error::RuntimePrerequisiteMissing)
            }
            _ => unreachable!(),
        }
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        unsafe {
            let ptr = cudnn_sys::cudnnGetErrorString(*self as _);
            let c_str = ffi::CStr::from_ptr(ptr);
            c_str.to_str().unwrap_or("[Non UTF8 description]")
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
