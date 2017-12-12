use std::os::raw::{c_float, c_int};

use cublas_sys;

macro_rules! cublas_fn {
    ($($type:ty),*) => {
        unsafe extern "C" fn(handle: ::cublas_sys::cublasHandle_t, $(_: $type),*) -> ::cublas_sys::cublasStatus_t
    };
}

pub trait Scalar {
    const IAMAX: cublas_fn!(c_int, *const Self, c_int, *mut c_int);
    const AXPY: cublas_fn!(c_int, *const Self, *const Self, c_int, *mut Self, c_int);
    const COPY: cublas_fn!(c_int, *const Self, c_int, *mut Self, c_int);
}

impl Scalar for c_float {
    const IAMAX: cublas_fn!(c_int, *const Self, c_int, *mut c_int) = cublas_sys::cublasIsamax_v2;
    const AXPY: cublas_fn!(c_int, *const Self, *const Self, c_int, *mut Self, c_int) = cublas_sys::cublasSaxpy_v2;
    const COPY: cublas_fn!(c_int, *const Self, c_int, *mut Self, c_int) = cublas_sys::cublasScopy_v2;
}
