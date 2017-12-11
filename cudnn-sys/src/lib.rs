#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

extern crate cuda_sys;
use cuda_sys::cudaStream_t;
use cuda_sys::libraryPropertyType;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
