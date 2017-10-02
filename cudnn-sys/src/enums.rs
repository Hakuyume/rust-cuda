#![allow(non_camel_case_types)]

#[repr(C)]
#[derive(Debug, PartialEq, Eq)]
pub enum cudnnDataType {
    CUDNN_DATA_FLOAT = 0,
    CUDNN_DATA_DOUBLE = 1,
    CUDNN_DATA_HALF = 2,
    CUDNN_DATA_INT8 = 3,
    CUDNN_DATA_INT32 = 4,
    CUDNN_DATA_INT8x4 = 5,
}

#[repr(C)]
pub enum cudnnDeterminism {
    CUDNN_NON_DETERMINISTIC = 0,
    CUDNN_DETERMINISTIC = 1,
}
