use {c_char, c_void, size_t};

use cudaError;
use cudaMemcpyKind;
use cudaStream;
use dim3;

#[link(name = "cudart")]
extern "system" {
    pub fn cudaGetErrorName(error: cudaError) -> *const c_char;
    pub fn cudaGetErrorString(error: cudaError) -> *const c_char;
    pub fn cudaFree(devPtr: *mut c_void) -> cudaError;
    pub fn cudaMalloc(devPtr: *mut *mut c_void, size: size_t) -> cudaError;
    pub fn cudaMemcpy(dst: *mut c_void,
                      src: *const c_void,
                      count: size_t,
                      kind: cudaMemcpyKind)
                      -> cudaError;
    pub fn cudaLaunchKernel(func: *const c_void,
                            gridDim: dim3,
                            blockDim: dim3,
                            args: *const *const c_void,
                            sharedMem: size_t,
                            stream: cudaStream)
                            -> cudaError;
}
