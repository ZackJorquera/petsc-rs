extern crate bindgen;
extern crate syn;

use std::env;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use quote::ToTokens;

// TODO: use types from rsmpi when i can over Petsc types
// from: https://github.com/rsmpi/rsmpi/blob/master/mpi-sys/src/rsmpi.h

// TODO: the `PetscScalar` type can be real or complex depending of compiler flags
// We should account for that in the code, right now it just uses reals
// https://petsc.org/release/docs/manualpages/Sys/PetscScalar.html#PetscScalar
// it is all compile time so what we have will work

fn main() {
    // TODO: get source and build petsc (idk, follow what rsmpi does maybe)
    // also look at libffi and how they do it with an external src

    let features = ["CARGO_FEATURE_PETSC_SCALAR_REAL_F64",
                    "CARGO_FEATURE_PETSC_SCALAR_COMPLEX_F64",
                    "CARGO_FEATURE_PETSC_SCALAR_REAL_F32",
                    "CARGO_FEATURE_PETSC_SCALAR_COMPLEX_F32",
                    "CARGO_FEATURE_PETSC_INT_I32",
                    "CARGO_FEATURE_PETSC_INT_I64"].iter()
        .map(|&x| env::var(x).ok().map(|o| if o == "1" { Some(x) } else { None }).flatten())
        .flatten().collect::<Vec<_>>();

    let scalar_features = features.iter().filter(|a| a.contains("PETSC_SCALAR_"))
        .copied().collect::<Vec<_>>();
    let int_features = features.iter().filter(|a| a.contains("PETSC_INT_"))
        .copied().collect::<Vec<_>>();
    
    assert!(scalar_features.len() == 1, "There must be exactly one \"petsc-scalar-*\" feature enabled");
    assert!(int_features.len() == 1, "There must be exactly one \"petsc-int-*\" feature enabled");

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

    // TODO: make better, show i do parsing. There must be a library that can do this
    // This gets the includes from mpicc
    if let Ok(output) = Command::new("mpicc").args(&["--show"]).output() {
        let gcc_command = String::from_utf8_lossy(&output.stdout).into_owned();
        let include_paths = gcc_command.split(' ').filter(|&s| &s[..2] == "-I");
        bindings = bindings.clang_args(include_paths);
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

        .allowlist_function("[A-Z][a-zA-Z]*")
        .allowlist_type("[A-Z][a-zA-Z]*")
        .allowlist_var("[A-Z][a-zA-Z]*")
        .allowlist_var("[A-Z_]*")

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

    // Assert we are using the right types

    // parse the bindings generated by bindgen
    let mut file = File::open(&bindings_file).unwrap();
    let mut content = String::new();
    file.read_to_string(&mut content).unwrap();
    let raw = syn::parse_file(&content).expect("Could not read generated bindings");

    // Find all variables named: PETSC_USE_*
    let petsc_use_idents = raw.items.iter()
        .filter_map(|item| match item {
            syn::Item::Const(c_item) => Some(format!("{}", c_item.ident.to_token_stream())),
            _ => None,
        }).filter(|ident| ident.contains("PETSC_USE_"))
        .collect::<Vec<_>>();

    // do asserts
    match scalar_features[0]
    {
        "CARGO_FEATURE_PETSC_SCALAR_REAL_F64" => assert!(petsc_use_idents.contains(&"PETSC_USE_REAL_DOUBLE".into()) &&
                                                         !petsc_use_idents.contains(&"PETSC_USE_COMPLEX".into()),
                                                         "PETSc is not compiled to use real `f64` for scalar"),
        "CARGO_FEATURE_PETSC_SCALAR_REAL_F32" => assert!(petsc_use_idents.contains(&"PETSC_USE_REAL_SINGLE".into()) &&
                                                         !petsc_use_idents.contains(&"PETSC_USE_COMPLEX".into()),
                                                         "PETSc is not compiled to use real `f32` for scalar"),
        "CARGO_FEATURE_PETSC_SCALAR_COMPLEX_F64" => assert!(petsc_use_idents.contains(&"PETSC_USE_REAL_DOUBLE".into()) &&
                                                            petsc_use_idents.contains(&"PETSC_USE_COMPLEX".into()),
                                                            "PETSc is not compiled to use complex `f64` for scalar"),
        "CARGO_FEATURE_PETSC_SCALAR_COMPLEX_F32" => assert!(petsc_use_idents.contains(&"PETSC_USE_REAL_DOUBLE".into()) &&
                                                            petsc_use_idents.contains(&"PETSC_USE_COMPLEX".into()),
                                                            "PETSc is not compiled to use complex `f32` for scalar"),
        _ => panic!("Invalid feature type for petsc scalar")
    }
    
    match int_features[0]
    {
        "CARGO_FEATURE_PETSC_INT_I64" => assert!(petsc_use_idents.contains(&"PETSC_USE_64BIT_INDICES".into()),
                                                 "PETSc is not compiled to use `i64` for ints"),
        "CARGO_FEATURE_PETSC_INT_I32" => assert!(!petsc_use_idents.contains(&"PETSC_USE_64BIT_INDICES".into()),
                                                 "PETSc is not compiled to use `i32` for ints"),
        _ => panic!("Invalid feature type for petsc int")
    }

    // TODO: allow different scalar and int types
    assert_eq!(scalar_features[0], "CARGO_FEATURE_PETSC_SCALAR_REAL_F64", "petsc-sys currently only supports using `f64` as `PetscScalar`. Please use the `petsc-scalar-real-f64` feature.");
    assert_eq!(int_features[0], "CARGO_FEATURE_PETSC_INT_I32", "petsc-sys currently only supports using `i32` as `PetscInt`. Please use the `petsc-int-i32` feature.");
}