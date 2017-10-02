#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Param4D {
    pub n: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub n_stride: usize,
    pub c_stride: usize,
    pub h_stride: usize,
    pub w_stride: usize,
}
