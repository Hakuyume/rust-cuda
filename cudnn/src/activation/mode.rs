use cudnn_sys;
use cudnn_sys::cudnnActivationMode::*;

#[derive(Clone, Copy, Debug)]
pub enum Mode {
    Sigmoid,
    Relu,
    Tanh,
    ClippedRelu,
    Elu,
}

impl Into<cudnn_sys::cudnnActivationMode> for Mode {
    fn into(self) -> cudnn_sys::cudnnActivationMode {
        match self {
            Mode::Sigmoid => CUDNN_ACTIVATION_SIGMOID,
            Mode::Relu => CUDNN_ACTIVATION_RELU,
            Mode::Tanh => CUDNN_ACTIVATION_TANH,
            Mode::ClippedRelu => CUDNN_ACTIVATION_CLIPPED_RELU,
            Mode::Elu => CUDNN_ACTIVATION_ELU,
        }
    }
}
