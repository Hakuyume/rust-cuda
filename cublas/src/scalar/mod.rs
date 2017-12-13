use std::os::raw::c_float;

use cublas_sys;

mod alias;
use self::alias::*;

pub trait Scalar {
    const IAMAX: Iamax<Self>;
    const ASUM: Asum<Self>;
    const AXPY: Axpy<Self>;
    const COPY: Copy<Self>;

    const GEMV: Gemv<Self>;
}

impl Scalar for c_float {
    const IAMAX: Iamax<Self> = cublas_sys::cublasIsamax_v2;
    const ASUM: Asum<Self> = cublas_sys::cublasSasum_v2;
    const AXPY: Axpy<Self> = cublas_sys::cublasSaxpy_v2;
    const COPY: Copy<Self> = cublas_sys::cublasScopy_v2;

    const GEMV: Gemv<Self> = cublas_sys::cublasSgemv_v2;
}
