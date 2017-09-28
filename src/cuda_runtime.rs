use {c_void, size_t};
use cudaError;
use cudaMemcpyKind;

#[link(name = "cudart")]
extern "C" {
    pub fn cudaMalloc(devPtr: *mut *mut c_void, size: size_t) -> cudaError;
    pub fn cudaFree(devPtr: *mut c_void) -> cudaError;
    pub fn cudaMemcpy(dst: *mut c_void,
                      src: *const c_void,
                      count: size_t,
                      kind: cudaMemcpyKind)
                      -> cudaError;
}
