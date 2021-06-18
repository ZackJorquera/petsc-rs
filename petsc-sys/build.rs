extern crate bindgen;
extern crate syn;

use std::env;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use quote::ToTokens;

fn main() {
    // TODO: get source and build petsc (idk, follow what rsmpi does maybe)
    // also look at libffi and how they do it with an external src

    let features = ["CARGO_FEATURE_PETSC_REAL_F64",
                    "CARGO_FEATURE_PETSC_REAL_F32",
                    "CARGO_FEATURE_PETSC_USE_COMPLEX",
                    "CARGO_FEATURE_PETSC_INT_I32",
                    "CARGO_FEATURE_PETSC_INT_I64"].iter()
        .map(|&x| env::var(x).ok().map(|o| if o == "1" { Some(x) } else { None }).flatten())
        .flatten().collect::<Vec<_>>();

    println!("cargo:rerun-if-env-changed=PETSC_DIR");
    println!("cargo:rerun-if-env-changed=PETSC_ARCH");
    println!("cargo:rerun-if-env-changed=PETSC_ARCH_RELEASE");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");

    let real_features = features.iter().filter(|a| a.contains("PETSC_REAL_"))
        .copied().collect::<Vec<_>>();
    let use_complex_feature = features.contains(&"CARGO_FEATURE_PETSC_USE_COMPLEX");
    let int_features = features.iter().filter(|a| a.contains("PETSC_INT_"))
        .copied().collect::<Vec<_>>();
    
    assert_eq!(real_features.len(),  1, 
        "There must be exactly one \"petsc-real-*\" feature enabled. There are {} enabled.",
            real_features.len());
    assert_eq!(int_features.len(), 1, 
        "There must be exactly one \"petsc-int-*\" feature enabled. There are {} enabled.",
            int_features.len());

    let profile = env::var("PROFILE").expect("No profile set.");

    // We do our best to find the PETSC install directory
    let petsc_dir = env::var("PETSC_DIR").map(|x| Path::new(&x).to_path_buf());
    let petsc_full_dir = petsc_dir.map(|petsc_dir| match profile.as_str() {
        "release" => match env::var("PETSC_ARCH_RELEASE") {
            Ok(arch) => petsc_dir.join(arch),
            Err(_) => match env::var("PETSC_ARCH") {
                Ok(arch) => petsc_dir.join(arch),
                Err(_) => petsc_dir,
            },
        }
        _ => match env::var("PETSC_ARCH") {
            Ok(arch) => petsc_dir.join(arch),
            Err(_) => petsc_dir,
        }
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

    let atleast_version = "3.15";
    let lib = match pkg_config::Config::new()
        .atleast_version(atleast_version)
        .probe("PETSc") { // note, this is case sensitive
            Ok(lib) => lib,
            Err(err) => { 
                eprintln!("Could not find library \'PETSc\', will try again. Error: {:?} ", err);
                match pkg_config::Config::new()
                    .atleast_version(atleast_version)
                    .probe("petsc") {
                        Ok(lib) => lib,
                        Err(err) => panic!("Could not find library \'petsc\', Error: {:?}", err)
                    }
            },
        };

    eprintln!("lib found: {:?}", lib);
    
    let mut bindings = bindgen::Builder::default();

    for dir in &lib.link_paths {
        println!("cargo:rustc-link-search={}", dir.display());
        // the binary will look for the petsc lib in the directory pointed at by LD_LIBRARY_PATH
        println!("cargo:rustc-env=LD_LIBRARY_PATH={}", dir.display());

        bindings = bindings.clang_arg(format!("-L{}", dir.to_string_lossy()));
    }
    for lib in &lib.libs {
        println!("cargo:rustc-link-lib={}", lib);

        // TODO: what does this do? it requires the crate libloading
        // bindings = bindings.dynamic_library_name(lib);
    }
    
    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=src/petsc_wrapper.h");

    for dir in &lib.include_paths {
        //println!("cargo:rerun-if-changed={}", dir.to_string_lossy());
        bindings = bindings.clang_arg(format!("-I{}", dir.to_string_lossy()));
    }

    // TODO: make better, show i do parsing. There must be a library that can do this
    // This gets the includes from mpicc
    // We could probably use the build-probe-mpi crate to get this info.
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

        .allowlist_function("[A-Z][a-zA-Z0-9]*")
        .allowlist_type("[A-Z][a-zA-Z0-9]*")
        .allowlist_var("[A-Z][a-zA-Z]*")
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
    match real_features[0]
    {
        "CARGO_FEATURE_PETSC_REAL_F64" => assert!(petsc_use_idents.contains(&"PETSC_USE_REAL_DOUBLE".into()),
            "PETSc is not compiled to use `f64` for real, but the feature \"petsc-real-f64\" is set."),
        "CARGO_FEATURE_PETSC_REAL_F32" => assert!(petsc_use_idents.contains(&"PETSC_USE_REAL_SINGLE".into()),
            "PETSc is not compiled to use `f32` for real, but the feature \"petsc-real-f32\" is set."),
        _ => panic!("Invalid feature type for petsc real")
    }

    if use_complex_feature {
        assert!(petsc_use_idents.contains(&"PETSC_USE_COMPLEX".into()),
                "PETSc is not compiled to use complex for scalar, but the feature \"petsc-use-complex\" is set.");
    } else {
        assert!(!petsc_use_idents.contains(&"PETSC_USE_COMPLEX".into()),
                "PETSc is compiled to use complex for scalar, but the feature \"petsc-use-complex\" is no set.");
    }
    
    match int_features[0]
    {
        "CARGO_FEATURE_PETSC_INT_I64" => assert!(petsc_use_idents.contains(&"PETSC_USE_64BIT_INDICES".into()),
            "PETSc is not compiled to use `i64` for ints, but the feature \"petsc-int-i64\" is set."),
        "CARGO_FEATURE_PETSC_INT_I32" => assert!(!petsc_use_idents.contains(&"PETSC_USE_64BIT_INDICES".into()),
            "PETSc is not compiled to use `i32` for ints, but the feature \"petsc-int-i32\" is set."),
        _ => panic!("Invalid feature type for petsc int")
    }
}