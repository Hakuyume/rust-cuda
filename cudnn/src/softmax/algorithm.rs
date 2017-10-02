use cudnn_sys;

#[derive(Clone, Copy, Debug)]
pub enum Algorithm {
    Fast,
    Accurate,
    Log,
}

impl Into<cudnn_sys::cudnnSoftmaxAlgorithm> for Algorithm {
    fn into(self) -> cudnn_sys::cudnnSoftmaxAlgorithm {
        match self {
            Algorithm::Fast => cudnn_sys::cudnnSoftmaxAlgorithm::CUDNN_SOFTMAX_FAST,
            Algorithm::Accurate => cudnn_sys::cudnnSoftmaxAlgorithm::CUDNN_SOFTMAX_ACCURATE,
            Algorithm::Log => cudnn_sys::cudnnSoftmaxAlgorithm::CUDNN_SOFTMAX_LOG,
        }
    }
}
