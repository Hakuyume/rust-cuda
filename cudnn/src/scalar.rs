use libc::c_float;

use cudnn_sys::cudnnDataType;

pub trait Scalar {
    const DATA_TYPE: cudnnDataType;
}

impl Scalar for c_float {
    const DATA_TYPE: cudnnDataType = cudnnDataType::CUDNN_DATA_FLOAT;
}

pub trait Float: Scalar {
    type Scale;
    fn into(self) -> Self::Scale;
}

impl Float for c_float {
    type Scale = c_float;
    fn into(self) -> c_float {
        self
    }
}
