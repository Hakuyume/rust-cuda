use std::os::raw::{c_float, c_int};

use cublas_sys;

macro_rules! cublas_fn {
    ($($type:ty),*) => {
        unsafe extern "C" fn(handle: ::cublas_sys::cublasHandle_t, $(_: $type),*) -> ::cublas_sys::cublasStatus_t
    };
}

type Iamax<T> = cublas_fn!(c_int, *const T, c_int, *mut c_int);
type Axpy<T> = cublas_fn!(c_int, *const T, *const T, c_int, *mut T, c_int);
type Copy<T> = cublas_fn!(c_int, *const T, c_int, *mut T, c_int);

pub trait Scalar {
    const IAMAX: Iamax<Self>;
    const AXPY: Axpy<Self>;
    const COPY: Copy<Self>;
}

impl Scalar for c_float {
    const IAMAX: Iamax<Self> = cublas_sys::cublasIsamax_v2;
    const AXPY: Axpy<Self> = cublas_sys::cublasSaxpy_v2;
    const COPY: Copy<Self> = cublas_sys::cublasScopy_v2;
}
