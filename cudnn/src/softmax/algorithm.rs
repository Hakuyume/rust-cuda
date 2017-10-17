use cudnn_sys;
use cudnn_sys::cudnnSoftmaxAlgorithm::*;

#[derive(Clone, Copy, Debug)]
pub enum Algorithm {
    Fast,
    Accurate,
    Log,
}

impl Into<cudnn_sys::cudnnSoftmaxAlgorithm> for Algorithm {
    fn into(self) -> cudnn_sys::cudnnSoftmaxAlgorithm {
        match self {
            Algorithm::Fast => CUDNN_SOFTMAX_FAST,
            Algorithm::Accurate => CUDNN_SOFTMAX_ACCURATE,
            Algorithm::Log => CUDNN_SOFTMAX_LOG,
        }
    }
}
