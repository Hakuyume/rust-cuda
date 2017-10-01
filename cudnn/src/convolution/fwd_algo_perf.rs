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
    fn from(fwd_algo_perf: cudnn_sys::cudnnConvolutionFwdAlgoPerf) -> FwdAlgoPerf {
        FwdAlgoPerf {
            algo: FwdAlgo::from(fwd_algo_perf.algo),
            status: error::wrap_status(fwd_algo_perf.status),
            time: fwd_algo_perf.time as f64,
            memory: fwd_algo_perf.memory as usize,
        }
    }
}
