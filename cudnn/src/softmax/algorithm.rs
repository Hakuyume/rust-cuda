use cudnn_sys;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Algorithm {
    Fast,
    Accurate,
    Log,
}

impl Into<cudnn_sys::cudnnSoftmaxAlgorithm_t> for Algorithm {
    fn into(self) -> cudnn_sys::cudnnSoftmaxAlgorithm_t {
        match self {
            Algorithm::Fast => cudnn_sys::cudnnSoftmaxAlgorithm_t_CUDNN_SOFTMAX_FAST,
            Algorithm::Accurate => cudnn_sys::cudnnSoftmaxAlgorithm_t_CUDNN_SOFTMAX_ACCURATE,
            Algorithm::Log => cudnn_sys::cudnnSoftmaxAlgorithm_t_CUDNN_SOFTMAX_LOG,
        }
    }
}
