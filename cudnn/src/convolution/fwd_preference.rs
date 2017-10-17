use cudnn_sys;
use cudnn_sys::cudnnConvolutionFwdPreference::*;

#[derive(Clone, Copy)]
pub enum FwdPreference {
    NoWorkspace,
    PreferFastest,
    SpecifyWorkspaceLimit(usize),
}

impl Into<(cudnn_sys::cudnnConvolutionFwdPreference, Option<usize>)> for FwdPreference {
    fn into(self) -> (cudnn_sys::cudnnConvolutionFwdPreference, Option<usize>) {
        match self {
            FwdPreference::NoWorkspace => (CUDNN_CONVOLUTION_FWD_NO_WORKSPACE, None),
            FwdPreference::PreferFastest => (CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, None),
            FwdPreference::SpecifyWorkspaceLimit(size) => {
                (CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, Some(size))
            }
        }
    }
}
