extern crate bindgen;

use std::env;
use std::path::Path;
use std::path::PathBuf;

// TODO: use types from rsmpi when i can over Petsc types
// from: https://github.com/rsmpi/rsmpi/blob/master/mpi-sys/src/rsmpi.h

// TODO: the `PetscScalar` type can be real or complex depending of compiler flags
// We should account for that in the code, right now it just uses reals
// https://petsc.org/release/docs/manualpages/Sys/PetscScalar.html#PetscScalar
// it is all compile time so what we have will work

fn main() {
    // TODO: get source and build petsc (idk, follow what rsmpi does maybe)
    // also look at libffi and how they do it with an external src

    // We do our best to find the PETSC install
    let petsc_dir = env::var("PETSC_DIR").map(|x| Path::new(&x).to_path_buf());
    let petsc_full_dir = petsc_dir.map(|petsc_dir| match env::var("PETSC_ARCH") {
        Ok(arch) => petsc_dir.join(arch),
        Err(_) => petsc_dir,
    });

    // In order for pkg_config to find the lib data for us we need the directory containing `PETSc.pc` 
    // to be in the env var PKG_CONFIG_PATH. For petsc we want `$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig`
    if let Ok(petsc_dir) = &petsc_full_dir {
        let pkgconfig_dir = petsc_dir.join("lib/pkgconfig");
        if let Some(path) = env::var_os("PKG_CONFIG_PATH") {
            let mut paths = env::split_paths(&path).collect::<Vec<_>>();
            paths.push(pkgconfig_dir);
            let new_path = env::join_paths(paths).unwrap();
            env::set_var("PKG_CONFIG_PATH", &new_path);
        } else {
            env::set_var("PKG_CONFIG_PATH", pkgconfig_dir);
        }
    }

    let lib = match pkg_config::Config::new()
        .atleast_version("3.10")
        .probe("PETSc") { // case sensitive, but it should have to be maybe TODO: allow for lower case
            Ok(lib) => lib,
            Err(err) => panic!("Could not find library \'PETSc\', Error: {:?}", err)
        };

    for dir in &lib.link_paths {
        println!("cargo:rustc-link-search={}", dir.display());
        // the binary will look for the petsc lib in the directory pointed at by LD_LIBRARY_PATH
        println!("cargo:rustc-env=LD_LIBRARY_PATH={}", dir.display());
    }
    for lib in &lib.libs {
        println!("cargo:rustc-link-lib={}", lib);
    }
    
    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=src/petsc_wrapper.h");

    let mut bindings = bindgen::Builder::default();

    for dir in &lib.include_paths {
        //println!("cargo:rerun-if-changed={}", dir.to_string_lossy());
        bindings = bindings.clang_arg(format!("-I{}", dir.to_string_lossy()));
    }

    let mpi_lib = match build_probe_mpi::probe() {
        Ok(lib) => lib,
        Err(errs) => {
            eprintln!("Could not find MPI library for various reasons:\n");
            for (i, err) in errs.iter().enumerate() {
                eprintln!("Reason #{}:\n{}\n", i, err);
            }
            panic!();
        }
    };

    for dir in &mpi_lib.lib_paths {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    for lib in &mpi_lib.libs {
        println!("cargo:rustc-link-lib={}", lib);
    }
    for dir in &mpi_lib.include_paths {
        bindings = bindings.clang_arg(format!("-I{}", dir.to_string_lossy()));
    }

    // TODO: get comments somehow
    // bindings = bindings.clang_arg("-fparse-all-comments").clang_arg("-fretain-comments-from-system-headers");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindings
        // The input header we would like to generate
        // bindings for.
        .header("src/petsc_wrapper.h")

        .allowlist_function("[A-Z][a-zA-Z0-9]*")
        .allowlist_type("[A-Z][a-zA-Z0-9]*")
        .allowlist_var("[A-Z][a-zA-Z0-9]*")
        .allowlist_var("[A-Z0-9_]*")

        .opaque_type("FILE")

        // There is no need to make bindings for mpi types as that has already been done in the mpi crate
        .blocklist_type("MPI\\w*")
        .blocklist_type("ompi\\w*")
        .blocklist_item("FP\\w*") // we need this because PETSc defines FP_* things twice and we will get errors
        .raw_line("use mpi::ffi::*;")

        // Tell cargo to not mangle the function names
        .trust_clang_mangling(false)
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Make C enums into rust enums not consts
        .default_enum_style(bindgen::EnumVariation::Rust{non_exhaustive:false})
        // Generate Comments
        .generate_comments(true)
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bindings_file = out_path.join("bindings.rs");
    bindings
        .write_to_file(&bindings_file)
        .expect("Couldn't write bindings!");

    // TODO: do some more fancy stuff
}