use std::os::raw::c_float;

use cudnn_sys;

pub trait Scalar {
    const DATA_TYPE: cudnn_sys::cudnnDataType_t;
}

impl Scalar for c_float {
    const DATA_TYPE: cudnn_sys::cudnnDataType_t = cudnn_sys::CUDNN_DATA_FLOAT;
}

pub trait Scale: Scalar {
    type Scale;
}

impl Scale for c_float {
    type Scale = c_float;
}
