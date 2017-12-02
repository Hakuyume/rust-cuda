use cudnn_sys::c_float;
use cudnn_sys::cudnnDataType;

pub trait Scalar {
    const DATA_TYPE: cudnnDataType;
}

impl Scalar for c_float {
    const DATA_TYPE: cudnnDataType = cudnnDataType::CUDNN_DATA_FLOAT;
}

pub trait Scale {
    type Scale;
}

impl Scale for c_float {
    type Scale = c_float;
}
