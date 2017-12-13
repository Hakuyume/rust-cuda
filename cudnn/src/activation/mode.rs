use cudnn_sys;

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum Mode {
    Sigmoid = cudnn_sys::CUDNN_ACTIVATION_SIGMOID,
    Relu = cudnn_sys::CUDNN_ACTIVATION_RELU,
    Tanh = cudnn_sys::CUDNN_ACTIVATION_TANH,
    ClippedRelu = cudnn_sys::CUDNN_ACTIVATION_CLIPPED_RELU,
    Elu = cudnn_sys::CUDNN_ACTIVATION_ELU,
}
