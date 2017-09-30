use cudnn_sys;

pub enum Mode {
    Instance,
    Channel,
}

impl Mode {
    pub fn as_raw(self) -> cudnn_sys::cudnnSoftmaxMode {
        match self {
            Mode::Instance => cudnn_sys::cudnnSoftmaxMode::CUDNN_SOFTMAX_MODE_INSTANCE,
            Mode::Channel => cudnn_sys::cudnnSoftmaxMode::CUDNN_SOFTMAX_MODE_CHANNEL,
        }
    }
}
