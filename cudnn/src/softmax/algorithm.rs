use cudnn_sys;

pub enum Algorithm {
    Fast,
    Accurate,
    Log,
}

impl Algorithm {
    pub fn as_raw(self) -> cudnn_sys::cudnnSoftmaxAlgorithm {
        match self {
            Algorithm::Fast => cudnn_sys::cudnnSoftmaxAlgorithm::CUDNN_SOFTMAX_FAST,
            Algorithm::Accurate => cudnn_sys::cudnnSoftmaxAlgorithm::CUDNN_SOFTMAX_ACCURATE,
            Algorithm::Log => cudnn_sys::cudnnSoftmaxAlgorithm::CUDNN_SOFTMAX_LOG,
        }
    }
}
