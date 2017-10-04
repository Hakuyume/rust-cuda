use std::env;
use std::path;
use std::process;

#[cfg(unix)]
pub fn compile_library(output: &str, files: &[&str]) {
    let out_dir = path::PathBuf::from(&env::var("OUT_DIR").unwrap());

    let mut objs = Vec::new();
    for file in files.iter() {
        let name = path::Path::new(file).file_stem().unwrap();
        let mut obj = name.to_owned();
        obj.push(".o");

        process::Command::new("nvcc")
            .args(&["-c", file, "-Xcompiler", "-fPIC", "-o"])
            .arg(&out_dir.join(&obj))
            .status()
            .unwrap();

        objs.push(obj);
    }

    process::Command::new("ar")
        .args(&["crus", output])
        .args(objs.iter())
        .current_dir(&out_dir)
        .status()
        .unwrap();

    println!("cargo:rustc-link-search={}", out_dir.display());
}
