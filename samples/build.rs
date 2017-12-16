extern crate nvcc;

fn main() {
    nvcc::compile_library("libkernel.a", &["src/bin/vector_add.cu"]);
}
