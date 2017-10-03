use cuda_sys;

pub struct Stream {
    stream: cuda_sys::cudaStream,
}

impl Stream {
    pub fn as_ptr(&self) -> cuda_sys::cudaStream {
        self.stream
    }

    pub fn as_mut_ptr(&mut self) -> cuda_sys::cudaStream {
        self.stream
    }
}
