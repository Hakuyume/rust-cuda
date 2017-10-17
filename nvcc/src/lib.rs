use std::env;
use std::path;
use std::process;

use std::error::Error;

macro_rules! fail {
    ($($e:expr),*) => {{
        eprintln!($($e,)*);
        panic!()
    }};
}

fn exec(name: &str, cmd: &mut process::Command) {
    match cmd.status() {
        Ok(status) => {
            if status.success() {
                ()
            } else {
                fail!("nvcc: \"{}\" exited with {}.", name, status);
            }
        }
        Err(err) => fail!("nvcc: Cannot execute \"{}\". {}", name, err.description()),
    }
}

#[cfg(unix)]
pub fn compile_library(output: &str, files: &[&str]) {
    assert!(output.starts_with("lib"));
    assert!(output.ends_with(".a"));
    let lib_name = &output[3..output.len() - 2];

    let out_dir = match env::var("OUT_DIR") {
        Ok(out_dir) => path::PathBuf::from(out_dir),
        Err(_) => fail!("nvcc: Cannot detect output directory."),
    };

    let mut objs = Vec::new();
    for file in files {
        let name = match path::Path::new(file).file_stem() {
            Some(name) => name,
            None => fail!("nvcc: Cannot detect stem for \"{}\".", file),
        };
        let mut obj = name.to_owned();
        obj.push(".o");

        let mut cmd = process::Command::new("nvcc");
        cmd.args(&["-c", file, "-Xcompiler", "-fPIC", "-o"])
            .arg(&out_dir.join(&obj));
        exec("nvcc", &mut cmd);

        objs.push(obj);
    }

    let mut cmd = process::Command::new("ar");
    cmd.args(&["crus", output])
        .args(&objs)
        .current_dir(&out_dir);
    exec("ar", &mut cmd);

    println!("cargo:rustc-link-lib=static={}", lib_name);
    println!("cargo:rustc-link-search=native={}", out_dir.display());
}
