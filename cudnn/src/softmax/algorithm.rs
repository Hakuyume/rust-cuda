use cudnn_sys;

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum Algorithm {
    Fast = cudnn_sys::CUDNN_SOFTMAX_FAST,
    Accurate = cudnn_sys::CUDNN_SOFTMAX_ACCURATE,
    Log = cudnn_sys::CUDNN_SOFTMAX_LOG,
}
