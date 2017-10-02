use std::error;
use std::ffi;
use std::fmt;
use std::result;

use cuda_sys;
use cuda_sys::cudaError;

#[derive(Clone, Copy, Debug)]
pub enum Error {
    MemoryAllocation,
    InitializationError,
    InvalidDevicePointer,
    Unknown,
}

pub trait TryFrom<T>: Sized {
    type Error;
    fn try_from(T) -> result::Result<Self, Self::Error>;
}

impl TryFrom<cudaError> for Error {
    type Error = ();
    fn try_from(value: cudaError) -> result::Result<Error, ()> {
        match value {
            cudaError::cudaSuccess => Err(()),
            cudaError::cudaErrorMemoryAllocation => Ok(Error::MemoryAllocation),
            cudaError::cudaErrorInitializationError => Ok(Error::InitializationError),
            cudaError::cudaErrorInvalidDevicePointer => Ok(Error::InvalidDevicePointer),
            cudaError::cudaErrorUnknown => Ok(Error::Unknown),
        }
    }
}

impl Into<cudaError> for Error {
    fn into(self) -> cudaError {
        match self {
            Error::MemoryAllocation => cudaError::cudaErrorMemoryAllocation,
            Error::InitializationError => cudaError::cudaErrorInitializationError,
            Error::InvalidDevicePointer => cudaError::cudaErrorInvalidDevicePointer,
            Error::Unknown => cudaError::cudaErrorUnknown,
        }
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        unsafe {
            let ptr = cuda_sys::cudaGetErrorString(Error::into(*self));
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
