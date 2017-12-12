extern crate bindgen;

use std::env;
use std::path;

fn main() {
    println!("cargo:rustc-link-lib=cublas");

    let bindings = bindgen::Builder::default()
        .header_contents("wrapper.h", "#include<cublas_v2.h>")
        .whitelist_function("cublas[A-Z].*")
        .whitelist_recursively(false)
        .whitelist_type("cublas[A-Z].*")
        .blacklist_type("cublasDataType_t")
        .whitelist_type("cu(?:Float|Double)?Complex")
        .no_convert_floats()
        .derive_debug(false)
        .prepend_enum_name(false)
        .generate()
        .unwrap();

    let out_path = path::PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .unwrap();
}
