#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

extern crate cuda_sys;
use cuda_sys::cudaStream_t;
use cuda_sys::cudaDataType;
use cuda_sys::libraryPropertyType;
use cuda_sys::{float2, double2};

pub use cuda_sys::cudaDataType as cublasDataType_t;
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
