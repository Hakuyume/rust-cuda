use cudnn_sys;

#[derive(Clone, Copy, Debug)]
pub enum Mode {
    Instance,
    Channel,
}

impl Into<cudnn_sys::cudnnSoftmaxMode_t> for Mode {
    fn into(self) -> cudnn_sys::cudnnSoftmaxMode_t {
        match self {
            Mode::Instance => cudnn_sys::CUDNN_SOFTMAX_MODE_INSTANCE,
            Mode::Channel => cudnn_sys::CUDNN_SOFTMAX_MODE_CHANNEL,
        }
    }
}
