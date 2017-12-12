extern crate bindgen;

use std::env;
use std::path;

fn main() {
    println!("cargo:rustc-link-lib=cudart");

    let bindings = bindgen::Builder::default()
        .header_contents("wrapper.h", "#include<cuda_runtime.h>")
        .whitelist_function("cuda[A-Z].*")
        .whitelist_type("cudaDataType")
        .whitelist_type("libraryPropertyType(?:_t)?")
        .whitelist_type("float2|double2")
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
