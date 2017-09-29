use std;

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
        match *self {
            Error::MemoryAllocation => "It was unable to allocate enough memory",
            Error::InitializationError => "The CUDA driver and runtime could not be initialized",
            Error::InvalidDevicePointer => "At least one device pointer is not a valid device pointer",
            Error::Unknown => "Unknown error",
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
