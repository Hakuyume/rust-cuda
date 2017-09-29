#![allow(non_camel_case_types)]

pub enum cudnnHandle {}

#[repr(C)]
pub enum cudnnStatus {
    CUDNN_STATUS_SUCCESS = 0,
    CUDNN_STATUS_NOT_INITIALIZED = 1,
    CUDNN_STATUS_ALLOC_FAILED = 2,
    CUDNN_STATUS_BAD_PARAM = 3,
    CUDNN_STATUS_INTERNAL_ERROR = 4,
    CUDNN_STATUS_INVALID_VALUE = 5,
    CUDNN_STATUS_ARCH_MISMATCH = 6,
    CUDNN_STATUS_MAPPING_ERROR = 7,
    CUDNN_STATUS_EXECUTION_FAILED = 8,
    CUDNN_STATUS_NOT_SUPPORTED = 9,
    CUDNN_STATUS_LICENSE_ERROR = 10,
    CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11,
}

#[link(name = "cudnn")]
extern "system" {
    cudnnCreate(handle: *mut *mut cudnnHandle) ->  cudnnStatus;
    cudnnDestroy(handle: *mut cudnnHandle) -> cudnnStatus;
}
