use cudnn_sys::c_float;
use cudnn_sys::cudnnDataType;

pub trait Scalar {
    const DATA_TYPE: cudnnDataType;
}

impl Scalar for c_float {
    const DATA_TYPE: cudnnDataType = cudnnDataType::CUDNN_DATA_FLOAT;
}

pub trait Float: Sized + Scalar {
    type Scale: From<Self>;
    const ZERO: Self;
    const ONE: Self;
}

impl Float for c_float {
    type Scale = c_float;
    const ZERO: c_float = 0.;
    const ONE: c_float = 1.;
}
