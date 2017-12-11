extern crate bindgen;

use std::env;
use std::path;

fn main() {
    println!("cargo:rustc-link-lib=cudart");

    let bindings = bindgen::Builder::default()
        .header_contents("wrapper.h", "#include<cuda_runtime.h>")
        .whitelist_function("cuda[A-Z].*")
        .whitelist_type("libraryPropertyType(?:_t)?")
        .generate()
        .unwrap();

    let out_path = path::PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .unwrap();
}
