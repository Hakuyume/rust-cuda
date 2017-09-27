use {c_void, size_t};
use cudaError_t;

#[link(name = "cudart")]
extern "C" {
    pub fn cudaMalloc(devPtr: *mut *mut c_void, size: size_t) -> cudaError_t;
    pub fn cudaFree(devPtr: *mut c_void) -> cudaError_t;
}
