use cudnn_sys;

#[derive(Clone, Copy, Debug)]
pub enum Mode {
    Sigmoid,
    Relu,
    Tanh,
    ClippedRelu,
    Elu,
}

impl Into<cudnn_sys::cudnnActivationMode_t> for Mode {
    fn into(self) -> cudnn_sys::cudnnActivationMode_t {
        match self {
            Mode::Sigmoid => cudnn_sys::cudnnActivationMode_t_CUDNN_ACTIVATION_SIGMOID,
            Mode::Relu => cudnn_sys::cudnnActivationMode_t_CUDNN_ACTIVATION_RELU,
            Mode::Tanh => cudnn_sys::cudnnActivationMode_t_CUDNN_ACTIVATION_TANH,
            Mode::ClippedRelu => cudnn_sys::cudnnActivationMode_t_CUDNN_ACTIVATION_CLIPPED_RELU,
            Mode::Elu => cudnn_sys::cudnnActivationMode_t_CUDNN_ACTIVATION_ELU,
        }
    }
}
