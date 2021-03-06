use cudnn_sys;

use error::TryFrom;
use Error;
use Result;

use super::FwdAlgo;

#[derive(Clone, Copy, Debug)]
pub struct FwdAlgoPerf {
    pub algo: FwdAlgo,
    pub status: Result<()>,
    pub time: f64,
    pub memory: usize,
}

impl From<cudnn_sys::cudnnConvolutionFwdAlgoPerf_t> for FwdAlgoPerf {
    fn from(value: cudnn_sys::cudnnConvolutionFwdAlgoPerf_t) -> FwdAlgoPerf {
        FwdAlgoPerf {
            algo: value.algo.into(),
            status: match Error::try_from(value.status) {
                Ok(err) => Err(err),
                Err(_) => Ok(()),
            },
            time: value.time as f64,
            memory: value.memory as usize,
        }
    }
}
