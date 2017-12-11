use cuda_sys;

pub struct Stream {
    stream: cuda_sys::cudaStream_t,
}

impl Stream {
    pub fn as_mut_ptr(&mut self) -> cuda_sys::cudaStream_t {
        self.stream
    }
}
