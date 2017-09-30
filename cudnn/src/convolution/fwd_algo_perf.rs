use cudnn_sys;

use super::FwdAlgo;

pub struct FwdAlgoPerf {
    pub algo: FwdAlgo,
    pub time: f64,
    pub memory: usize,
}

impl FwdAlgoPerf {
    pub fn from_raw(fwd_algo_perf: cudnn_sys::cudnnConvolutionFwdAlgoPerf) -> FwdAlgoPerf {
        FwdAlgoPerf {
            algo: FwdAlgo::from_raw(fwd_algo_perf.algo),
            time: fwd_algo_perf.time as f64,
            memory: fwd_algo_perf.memory as usize,
        }
    }
}
