use std::os::raw::c_int;

use cublas_sys;

macro_rules! cublas_fn {
    ($($type:ty),*) => {
        unsafe extern "C" fn(handle: cublas_sys::cublasHandle_t, $(_: $type),*) -> cublas_sys::cublasStatus_t
    };
}

pub type Iamax<T> = cublas_fn!(c_int, *const T, c_int, *mut c_int);
pub type Axpy<T> = cublas_fn!(c_int, *const T, *const T, c_int, *mut T, c_int);
pub type Copy<T> = cublas_fn!(c_int, *const T, c_int, *mut T, c_int);

pub type Gemv<T> = cublas_fn!(cublas_sys::cublasOperation_t, c_int, c_int,
                              *const T,
                              *const T, c_int,
                              *const T, c_int,
                              *const T,
                              *mut T, c_int);
