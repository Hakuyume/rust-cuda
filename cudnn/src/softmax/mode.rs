use cudnn_sys;
use cudnn_sys::cudnnSoftmaxMode::*;

#[derive(Clone, Copy, Debug)]
pub enum Mode {
    Instance,
    Channel,
}

impl Into<cudnn_sys::cudnnSoftmaxMode> for Mode {
    fn into(self) -> cudnn_sys::cudnnSoftmaxMode {
        match self {
            Mode::Instance => CUDNN_SOFTMAX_MODE_INSTANCE,
            Mode::Channel => CUDNN_SOFTMAX_MODE_CHANNEL,
        }
    }
}
