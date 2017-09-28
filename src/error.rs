use std;
use cuda_sys::cudaError_t;

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

impl From<cudaError_t> for Error {
    fn from(error: cudaError_t) -> Self {
        match error {
            cudaError_t::cudaErrorMemoryAllocation => Error::MemoryAllocation,
            cudaError_t::cudaErrorInitializationError => Error::InitializationError,
            cudaError_t::cudaErrorInvalidDevicePointer => Error::InvalidDevicePointer,
            _ => Error::Unknown,
        }
    }
}
