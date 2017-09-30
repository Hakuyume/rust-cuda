use cuda::memory;

use cudnn_sys;
use cudnn_sys::c_void;

use scalar;
use context;
use tensor;
use Result;

pub enum Algorithm {
    Fast,
    Accurate,
    Log,
}

pub enum Mode {
    Instance,
    Channel,
}

pub fn forward<T: scalar::Float>(context: &context::Context,
                                 algo: Algorithm,
                                 mode: Mode,
                                 alpha: T,
                                 x: &memory::Slice<T>,
                                 x_desc: &tensor::TensorDescriptor<T>,
                                 beta: T,
                                 y: &mut memory::Slice<T>,
                                 y_desc: &tensor::TensorDescriptor<T>)
                                 -> Result<()> {
    assert_eq!(x.len(), x_desc.len());
    assert_eq!(y.len(), y_desc.len());

    let algo = match algo {
        Algorithm::Fast => cudnn_sys::cudnnSoftmaxAlgorithm::CUDNN_SOFTMAX_FAST,
        Algorithm::Accurate => cudnn_sys::cudnnSoftmaxAlgorithm::CUDNN_SOFTMAX_ACCURATE,
        Algorithm::Log => cudnn_sys::cudnnSoftmaxAlgorithm::CUDNN_SOFTMAX_LOG,
    };
    let mode = match mode {
        Mode::Instance => cudnn_sys::cudnnSoftmaxMode::CUDNN_SOFTMAX_MODE_INSTANCE,
        Mode::Channel => cudnn_sys::cudnnSoftmaxMode::CUDNN_SOFTMAX_MODE_CHANNEL,
    };

    let params: &[T::Scale] = &[alpha.into(), beta.into()];

    unsafe {
        try_call!(cudnn_sys::cudnnSoftmaxForward(context.as_raw(),
                                                 algo,
                                                 mode,
                                                 &params[0] as *const T::Scale as *const c_void,
                                                 x_desc.as_raw(),
                                                 x.as_ptr() as *const c_void,
                                                 &params[1] as *const T::Scale as *const c_void,
                                                 y_desc.as_raw(),
                                                 y.as_mut_ptr() as *mut c_void))
    }
    Ok(())
}
