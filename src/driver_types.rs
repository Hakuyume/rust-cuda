#![allow(non_camel_case_types)]

#[repr(C)]
pub enum cudaError {
    cudaSuccess = 0,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorInvalidDevicePointer = 17,
}
