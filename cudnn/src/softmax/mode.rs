use cudnn_sys;

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum Mode {
    Instance = cudnn_sys::CUDNN_SOFTMAX_MODE_INSTANCE,
    Channel = cudnn_sys::CUDNN_SOFTMAX_MODE_CHANNEL,
}
