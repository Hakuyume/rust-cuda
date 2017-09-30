use std;

use cudnn_sys;
use cudnn_sys::cudnnStatus;

use Result;

#[derive(Debug)]
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

impl std::error::Error for Error {
    fn description(&self) -> &str {
        let status = match *self {
            Error::NotInitialized => cudnnStatus::CUDNN_STATUS_NOT_INITIALIZED,
            Error::AllocFailed => cudnnStatus::CUDNN_STATUS_ALLOC_FAILED,
            Error::BadParam => cudnnStatus::CUDNN_STATUS_BAD_PARAM,
            Error::InternalError => cudnnStatus::CUDNN_STATUS_INTERNAL_ERROR,
            Error::InvalidValue => cudnnStatus::CUDNN_STATUS_INVALID_VALUE,
            Error::ArchMismatch => cudnnStatus::CUDNN_STATUS_ARCH_MISMATCH,
            Error::MappingError => cudnnStatus::CUDNN_STATUS_MAPPING_ERROR,
            Error::ExecutionFailed => cudnnStatus::CUDNN_STATUS_EXECUTION_FAILED,
            Error::NotSupported => cudnnStatus::CUDNN_STATUS_NOT_SUPPORTED,
            Error::LicenseError => cudnnStatus::CUDNN_STATUS_LICENSE_ERROR,
            Error::RuntimePrerequisiteMissing => {
                cudnnStatus::CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING
            }
        };
        unsafe {
            let ptr = cudnn_sys::cudnnGetErrorString(status);
            let c_str = std::ffi::CStr::from_ptr(ptr);
            c_str.to_str().unwrap_or("[Non UTF8 description]")
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub fn wrap_status(status: cudnnStatus) -> Result<()> {
    match status {
        cudnnStatus::CUDNN_STATUS_SUCCESS => Ok(()),
        cudnnStatus::CUDNN_STATUS_NOT_INITIALIZED => Err(Error::NotInitialized),
        cudnnStatus::CUDNN_STATUS_ALLOC_FAILED => Err(Error::AllocFailed),
        cudnnStatus::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam),
        cudnnStatus::CUDNN_STATUS_INTERNAL_ERROR => Err(Error::InternalError),
        cudnnStatus::CUDNN_STATUS_INVALID_VALUE => Err(Error::InvalidValue),
        cudnnStatus::CUDNN_STATUS_ARCH_MISMATCH => Err(Error::ArchMismatch),
        cudnnStatus::CUDNN_STATUS_MAPPING_ERROR => Err(Error::MappingError),
        cudnnStatus::CUDNN_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed),
        cudnnStatus::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported),
        cudnnStatus::CUDNN_STATUS_LICENSE_ERROR => Err(Error::LicenseError),
        cudnnStatus::CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING => {
            Err(Error::RuntimePrerequisiteMissing)
        }
    }
}

macro_rules! try_call {
    ($call:expr) => {try!(::error::wrap_status($call))};
}
