use std::env;
use std::error::Error;
use std::path;
use std::process;

#[cfg(unix)]
pub fn compile_library(output: &str, files: &[&str]) {
    let out_dir = match env::var("OUT_DIR") {
        Ok(out_dir) => path::PathBuf::from(out_dir),
        Err(_) => {
            eprintln!("nvcc: Cannot detect output directory.");
            panic!()
        }
    };

    let mut objs = Vec::new();
    for file in files.iter() {
        let name = match path::Path::new(file).file_stem() {
            Some(name) => name,
            None => {
                eprintln!("nvcc: Cannot detect stem for \"{}\".", file);
                panic!()
            }
        };
        let mut obj = name.to_owned();
        obj.push(".o");

        let status = process::Command::new("nvcc")
            .args(&["-c", file, "-Xcompiler", "-fPIC", "-o"])
            .arg(&out_dir.join(&obj))
            .status();
        match status {
            Ok(status) => {
                if status.success() {
                    ()
                } else {
                    eprintln!("nvcc: \"nvcc\" exited with {}.", status);
                    panic!()
                }
            }
            Err(err) => {
                eprintln!("nvcc: Cannot execute \"nvcc\". {}", err.description());
                panic!()
            }
        }

        objs.push(obj);
    }

    let status = process::Command::new("ar")
        .args(&["crus", output])
        .args(objs.iter())
        .current_dir(&out_dir)
        .status();
    match status {
        Ok(status) => {
            if status.success() {
                ()
            } else {
                eprintln!("nvcc: \"ar\" exited with {}.", status);
                panic!()
            }
        }
        Err(err) => {
            eprintln!("nvcc: Cannot execute \"ar\". {}", err.description());
            panic!()
        }
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
}
