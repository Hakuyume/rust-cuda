use std::error;
use std::ffi;
use std::fmt;
use std::result;

use cudnn_sys;
use cudnn_sys::cudnnStatus::*;

#[derive(Clone, Copy, Debug)]
pub enum Error {
    NotInitialized,
    AllocFailed,
    BadParam,
    InternalError,
    InvalidValue,
    ArchMismatch,
    MappingError,
    ExecutionFailed,
    NotSupported,
    LicenseError,
    RuntimePrerequisiteMissing,
}

pub trait TryFrom<T>: Sized {
    type Error;
    fn try_from(T) -> result::Result<Self, Self::Error>;
}

impl TryFrom<cudnn_sys::cudnnStatus> for Error {
    type Error = ();
    fn try_from(value: cudnn_sys::cudnnStatus) -> result::Result<Error, ()> {
        match value {
            CUDNN_STATUS_SUCCESS => Err(()),
            CUDNN_STATUS_NOT_INITIALIZED => Ok(Error::NotInitialized),
            CUDNN_STATUS_ALLOC_FAILED => Ok(Error::AllocFailed),
            CUDNN_STATUS_BAD_PARAM => Ok(Error::BadParam),
            CUDNN_STATUS_INTERNAL_ERROR => Ok(Error::InternalError),
            CUDNN_STATUS_INVALID_VALUE => Ok(Error::InvalidValue),
            CUDNN_STATUS_ARCH_MISMATCH => Ok(Error::ArchMismatch),
            CUDNN_STATUS_MAPPING_ERROR => Ok(Error::MappingError),
            CUDNN_STATUS_EXECUTION_FAILED => Ok(Error::ExecutionFailed),
            CUDNN_STATUS_NOT_SUPPORTED => Ok(Error::NotSupported),
            CUDNN_STATUS_LICENSE_ERROR => Ok(Error::LicenseError),
            CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING => Ok(Error::RuntimePrerequisiteMissing),
        }
    }
}

impl Into<cudnn_sys::cudnnStatus> for Error {
    fn into(self) -> cudnn_sys::cudnnStatus {
        match self {
            Error::NotInitialized => CUDNN_STATUS_NOT_INITIALIZED,
            Error::AllocFailed => CUDNN_STATUS_ALLOC_FAILED,
            Error::BadParam => CUDNN_STATUS_BAD_PARAM,
            Error::InternalError => CUDNN_STATUS_INTERNAL_ERROR,
            Error::InvalidValue => CUDNN_STATUS_INVALID_VALUE,
            Error::ArchMismatch => CUDNN_STATUS_ARCH_MISMATCH,
            Error::MappingError => CUDNN_STATUS_MAPPING_ERROR,
            Error::ExecutionFailed => CUDNN_STATUS_EXECUTION_FAILED,
            Error::NotSupported => CUDNN_STATUS_NOT_SUPPORTED,
            Error::LicenseError => CUDNN_STATUS_LICENSE_ERROR,
            Error::RuntimePrerequisiteMissing => CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING,
        }
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        unsafe {
            let ptr = cudnn_sys::cudnnGetErrorString(self.clone().into());
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
