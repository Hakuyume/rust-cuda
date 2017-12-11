use cudnn_sys;

#[derive(Clone, Copy)]
pub enum FwdPreference {
    NoWorkspace,
    PreferFastest,
    SpecifyWorkspaceLimit(usize),
}

impl Into<(cudnn_sys::cudnnConvolutionFwdPreference_t, Option<usize>)> for FwdPreference {
    fn into(self) -> (cudnn_sys::cudnnConvolutionFwdPreference_t, Option<usize>) {
        match self {
            FwdPreference::NoWorkspace => {
                (cudnn_sys::cudnnConvolutionFwdPreference_t_CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
                 None)
            }
            FwdPreference::PreferFastest => {
                (cudnn_sys::cudnnConvolutionFwdPreference_t_CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                 None)
            }
            FwdPreference::SpecifyWorkspaceLimit(size) => {
                (cudnn_sys::cudnnConvolutionFwdPreference_t_CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, Some(size))
            }
        }
    }
}
