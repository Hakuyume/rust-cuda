use cudnn_sys;

#[derive(Clone, Copy)]
pub enum BwdFilterPreference {
    NoWorkspace,
    PreferFastest,
    SpecifyWorkspaceLimit(usize),
}

impl Into<(cudnn_sys::cudnnConvolutionBwdFilterPreference_t, Option<usize>)>
    for BwdFilterPreference {
    fn into(self) -> (cudnn_sys::cudnnConvolutionBwdFilterPreference_t, Option<usize>) {
        match self {
            BwdFilterPreference::NoWorkspace => {
                (cudnn_sys::cudnnConvolutionBwdFilterPreference_t_CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE, None)
            }
            BwdFilterPreference::PreferFastest => {
                (cudnn_sys::cudnnConvolutionBwdFilterPreference_t_CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, None)
            }
            BwdFilterPreference::SpecifyWorkspaceLimit(size) => {
                (cudnn_sys::cudnnConvolutionBwdFilterPreference_t_CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, Some(size))
            }
        }
    }
}
