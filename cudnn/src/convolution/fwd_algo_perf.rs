use cudnn_sys;

use error;
use Result;

use super::FwdAlgo;

#[derive(Clone, Copy, Debug)]
pub struct FwdAlgoPerf {
    pub algo: FwdAlgo,
    pub status: Result<()>,
    pub time: f64,
    pub memory: usize,
}

impl From<cudnn_sys::cudnnConvolutionFwdAlgoPerf> for FwdAlgoPerf {
    fn from(value: cudnn_sys::cudnnConvolutionFwdAlgoPerf) -> FwdAlgoPerf {
        FwdAlgoPerf {
            algo: FwdAlgo::from(value.algo),
            status: error::wrap_status(value.status),
            time: value.time as f64,
            memory: value.memory as usize,
        }
    }
}
