use cudnn_sys;
use cudnn_sys::cudnnConvolutionBwdFilterPreference::*;

#[derive(Clone, Copy)]
pub enum BwdFilterPreference {
    NoWorkspace,
    PreferFastest,
    SpecifyWorkspaceLimit(usize),
}

impl Into<(cudnn_sys::cudnnConvolutionBwdFilterPreference, Option<usize>)> for BwdFilterPreference {
    fn into(self) -> (cudnn_sys::cudnnConvolutionBwdFilterPreference, Option<usize>) {
        match self {
            BwdFilterPreference::NoWorkspace => (CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE, None),
            BwdFilterPreference::PreferFastest => {
                (CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, None)
            }
            BwdFilterPreference::SpecifyWorkspaceLimit(size) => {
                (CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, Some(size))
            }
        }
    }
}
