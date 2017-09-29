use std;

use cuda_sys;
use cuda_sys::cudaError;

use Result;

#[derive(Debug)]
pub enum Error {
    MemoryAllocation,
    InitializationError,
    InvalidDevicePointer,
    Unknown,
}

impl std::error::Error for Error {
    fn description(&self) -> &str {
        let err = match *self {
            Error::MemoryAllocation => cudaError::cudaErrorMemoryAllocation,
            Error::InitializationError => cudaError::cudaErrorInitializationError,
            Error::InvalidDevicePointer => cudaError::cudaErrorInvalidDevicePointer,
            Error::Unknown => cudaError::cudaErrorUnknown,
        };
        unsafe {
            let ptr = cuda_sys::cudaGetErrorString(err);
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

pub fn wrap_error(err: cudaError) -> Result<()> {
    match err {
        cudaError::cudaSuccess => Ok(()),
        cudaError::cudaErrorMemoryAllocation => Err(Error::MemoryAllocation),
        cudaError::cudaErrorInitializationError => Err(Error::InitializationError),
        cudaError::cudaErrorInvalidDevicePointer => Err(Error::InvalidDevicePointer),
        cudaError::cudaErrorUnknown => Err(Error::Unknown),
    }
}

#[macro_export]
macro_rules! safe_call {
    ($call:expr) => {try!($crate::error::wrap_error($call))};
}
