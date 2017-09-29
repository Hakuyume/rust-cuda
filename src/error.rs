use std;

use cuda_sys::cudaError;

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

impl From<cudaError> for Error {
    fn from(error: cudaError) -> Self {
        match error {
            cudaError::cudaErrorMemoryAllocation => Error::MemoryAllocation,
            cudaError::cudaErrorInitializationError => Error::InitializationError,
            cudaError::cudaErrorInvalidDevicePointer => Error::InvalidDevicePointer,
            _ => Error::Unknown,
        }
    }
}
