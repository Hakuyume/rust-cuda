extern crate bindgen;

use std::env;
use std::path;

fn main() {
    println!("cargo:rustc-link-lib=cudnn");

    let bindings = bindgen::Builder::default()
        .header_contents("wrapper.h", "#include<cudnn.h>")
        .whitelist_function("cudnn[A-Z].*")
        .whitelist_recursively(false)
        .whitelist_type("cudnn[A-Z].*")
        .generate()
        .unwrap();

    let out_path = path::PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .unwrap();
}
