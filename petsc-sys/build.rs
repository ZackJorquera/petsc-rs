extern crate bindgen;
extern crate syn;

use std::borrow::Cow;
use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;

use quote::{ToTokens, quote};
use proc_macro2::{Ident, Span};
use syn::ItemConst;
// use semver::Version;

fn rustfmt_string(code_str: &str) -> Cow<'_, str> {
    if let Ok(mut child) = Command::new("rustfmt")
        .arg("--emit=stdout")
        .arg("--edition=2018")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        child.stdin.as_mut().unwrap().write_all(code_str.as_bytes()).unwrap();
        
        if let Ok(output) = child.wait_with_output() {
            if output.status.success() {
                return Cow::Owned(String::from_utf8(output.stdout).unwrap());
            }
        }
    }
    Cow::Borrowed(code_str)
}

fn create_enum_from_consts(name: Ident, items: Vec<ItemConst>, repr_type: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let fn_ident = Ident::new(&format!("create_enum_from_consts_test_layout_{}", name), Span::call_site());
    let item_idents = items.into_iter().map(|i| i.ident);
    let item_idents2 = item_idents.clone();
    quote! {
        #[repr(#repr_type)]
        #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
        pub enum #name {
            #(
                #item_idents = #item_idents as #repr_type,
            )*
        }

        #[test]
        fn #fn_ident() {
            assert_eq!(
                ::std::mem::size_of::<#name>(),
                ::std::mem::size_of::<#repr_type>(),
                concat!("Size of: ", stringify!(#name))
            );
            assert_eq!(
                ::std::mem::align_of::<#name>(),
                ::std::mem::align_of::<#repr_type>(),
                concat!("Alignment of ", stringify!(#name))
            );
            #(
                assert_eq!(
                    unsafe { ::std::mem::transmute::<#name, #repr_type>(#name::#item_idents2) },
                    #item_idents2 as #repr_type,
                    concat!("Value of: ", stringify!(#name::#item_idents2))
                );
            )*
        }
    }
}

fn create_type_enum_and_table(name: Ident, items: Vec<ItemConst>) -> proc_macro2::TokenStream {
    let enum_ident = Ident::new(&format!("{}Enum", name), Span::call_site());
    let table_ident = Ident::new(&format!("{}_TABLE", name).to_uppercase(), Span::call_site());
    let fn_ident = Ident::new(&format!("create_type_enum_and_table_test_values_{}", name), Span::call_site());
    let item_idents = items.into_iter().map(|i| i.ident).collect::<Vec<_>>();
    let i = 0usize..item_idents.len();
    let item_idents2 = item_idents.clone();
    quote! {
        #[repr(usize)]
        #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
        pub enum #enum_ident {
            #(
                #item_idents = #i,
            )*
        }

        pub static #table_ident: &'static [&'static [u8]] = &[
            #(
                #item_idents,
            )*
        ];

        #[test]
        fn #fn_ident() {
            #(
                assert_eq!(
                    #table_ident[#enum_ident::#item_idents2 as usize],
                    #item_idents2,
                    concat!("Value of: ", stringify!(#enum_ident::#item_idents2))
                );
            )*
        }
    }
}

fn create_all_type_enums(consts: &Vec<ItemConst>) -> proc_macro2::TokenStream {
    // I think these are correct. They might miss a few or grab to many
    // but i think it is better than manually grabbing everything.
    let enum_ident_strs = &[
        r"MatType",
        r"DMType",
        r"PCType",
        r"KSPType",
        r"PetscSpaceType",
        r"PetscDualSpaceType",
    ];
    // These work in pairs, the first is something it must match
    // the second is something it cant match. This array MUST be
    // exactly 2 times as long as `enum_ident_strs`.
    let regex_pats = &[
        r"^MAT[A-Z0-9]+$",
        r"^MAT(SOLVE|PRODUCT|ORDERING|COLORING|PARTITIONING|SEQUSFFT)[A-Z0-9]*$",
        r"^DM[A-Z0-9]+$",
        r"^DM(FIELD)[A-Z0-9]*$",
        r"^PC[A-Z0-9]+$",
        r"^PC(GAMG[AGC])[A-Z0-9]*$",
        r"^KSP[A-Z0-9]+$",
        r"^PC(GUESS)[A-Z0-9]*$",
        r"^PETSCSPACE[A-Z0-9]+$",
        r"^PETSCSPACE$", // Nothing should match this
        r"^PETSCDUALSPACE[A-Z0-9]+$",
        r"^PETSCDUALSPACE$", // Nothing should match this
    ];
    assert_eq!(enum_ident_strs.len() * 2, regex_pats.len());

    let token_streams = enum_ident_strs.iter().zip(regex_pats.chunks_exact(2)).map(|(&name, pats)| {
        let ident = Ident::new(name, Span::call_site());
        let re1 = regex::Regex::new(pats[0]).unwrap();
        let re2 = regex::Regex::new(pats[1]).unwrap();
        let consts_for_enum = consts.iter().filter_map(|c| {
            let s = format!("{}", c.ident.to_token_stream());
            if re1.is_match(&s) && !re2.is_match(&s) {
                Some(c.clone())
            } else {
                None
            }
        });
        
        create_type_enum_and_table(ident, consts_for_enum.collect())
    });

    quote! {
        #(
            #token_streams
        )*
    }
}

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

    // let profile = env::var("PROFILE").expect("No profile set.");

    let petsc_lib = build_probe_petsc::probe(None);
    let lib = &petsc_lib.lib;

    // let lib_version = Version::parse(&lib.version).unwrap();
    // let header_version = petsc_lib.get_version_from_consts();
    // eprintln!("lib found: {:?}, lib version: {:?} header version: {:?}", lib, lib_version, header_version);
    
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

    // Assert we are using the right types

    // parse the bindings generated by bindgen
    let mut file = File::open(&bindings_file).unwrap();
    let mut content = String::new();
    file.read_to_string(&mut content).unwrap();
    let raw = syn::parse_file(&content).expect("Could not read generated bindings");

    let raw_const_items = raw.items.iter()
        .filter_map(|item| match item {
            syn::Item::Const(c_item) => Some(c_item.clone()),
            _ => None,
        }).collect::<Vec<_>>();

    // Find all variables named: PETSC_ERR_*
    let petsc_err_consts = raw_const_items.iter()
        .filter_map(|c_item| if format!("{}", c_item.ident.to_token_stream()).contains("PETSC_ERR_") {
            Some(c_item.clone())
        } else {
            None
        }).collect::<Vec<_>>();
    
    let enum_file = out_path.join("enums.rs");
    let mut f = File::create(enum_file).unwrap();
    // we want i32 because `PetscErrorCode` is i32 (or really it is c_int)
    let code_string = format!("{}\n{}", 
        create_enum_from_consts(Ident::new("PetscErrorCodeEnum", Span::call_site()), petsc_err_consts, quote!{i32}).into_token_stream(),
        create_all_type_enums(&raw_const_items).into_token_stream());
    // TODO: we should format the code
    f.write(rustfmt_string(&code_string).as_bytes()).unwrap();

    // do asserts
    match real_features[0]
    {
        "CARGO_FEATURE_PETSC_REAL_F64" => assert!(petsc_lib.consts_contains("PETSC_USE_REAL_DOUBLE"),
            "PETSc is not compiled to use `f64` for real, but the feature \"petsc-real-f64\" is set."),
        "CARGO_FEATURE_PETSC_REAL_F32" => assert!(petsc_lib.consts_contains("PETSC_USE_REAL_SINGLE"),
            "PETSc is not compiled to use `f32` for real, but the feature \"petsc-real-f32\" is set."),
        _ => panic!("Invalid feature type for petsc real")
    }

    if use_complex_feature {
        assert!(petsc_lib.consts_contains("PETSC_USE_COMPLEX"),
                "PETSc is not compiled to use complex for scalar, but the feature \"petsc-use-complex\" is set.");

        panic!("Using complex numbers as PetscScalar is currently not available. Please disable \"petsc-use-complex\".");
    } else {
        assert!(!petsc_lib.consts_contains(&"PETSC_USE_COMPLEX"),
                "PETSc is compiled to use complex for scalar, but the feature \"petsc-use-complex\" is no set.");
    }
    
    match int_features[0]
    {
        "CARGO_FEATURE_PETSC_INT_I64" => assert!(petsc_lib.consts_contains("PETSC_USE_64BIT_INDICES"),
            "PETSc is not compiled to use `i64` for ints, but the feature \"petsc-int-i64\" is set."),
        "CARGO_FEATURE_PETSC_INT_I32" => assert!(!petsc_lib.consts_contains("PETSC_USE_64BIT_INDICES"),
            "PETSc is not compiled to use `i32` for ints, but the feature \"petsc-int-i32\" is set."),
        _ => panic!("Invalid feature type for petsc int")
    }
}