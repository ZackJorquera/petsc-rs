extern crate bindgen;
extern crate syn;

use std::env;
use std::path::Path;

use quote::ToTokens;
use regex::Regex;
use semver::{Prerelease, Version};

pub static CONSTS_TO_GET_REGEX: [&str; 2] = [
    "PETSC_USE_.+",
    "PETSC_VERSION_.+",
];

pub struct PetscProber {
    pub lib: pkg_config::Library,
    pub build_data: Vec<(String, String)>,
}

pub fn probe<'a>(atleast_version: impl Into<Option<&'a str>>) -> PetscProber {
    println!("cargo:rerun-if-env-changed=PETSC_DIR");
    println!("cargo:rerun-if-env-changed=PETSC_ARCH");
    println!("cargo:rerun-if-env-changed=PETSC_ARCH_RELEASE");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");

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

    let atleast_version = atleast_version.into();
    let lib = match { let mut cfg = pkg_config::Config::new();
        atleast_version.clone().map(|ver| cfg.atleast_version(ver));
        cfg.probe("PETSc") } { // note, this is case sensitive
            Ok(lib) => lib,
            Err(err) => { 
                eprintln!("Could not find library \'PETSc\', will try again. Error: {:?} ", err);
                match { let mut cfg = pkg_config::Config::new();
                    atleast_version.map(|ver| cfg.atleast_version(ver));
                    cfg.probe("petsc") } {
                        Ok(lib) => lib,
                        Err(err) => panic!("Could not find library \'petsc\', Error: {:?}", err)
                    }
            },
        };

    eprintln!("lib found: {:?}", lib);

    let mut bindings = bindgen::Builder::default();

    for dir in &lib.include_paths {
        println!("cargo:rerun-if-changed={}", dir.to_string_lossy());
        bindings = bindings.clang_arg(format!("-I{}", dir.to_string_lossy()));
    }

    let bindings = bindings
        // The input header we would like to generate
        // bindings for.
        .header_contents("phantom_header.h", "#include <petscconf.h>\n#include <petscversion.h>\n")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Generate Comments
        .generate_comments(true)
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    let out_string = bindings.to_string();
    
    let raw = syn::parse_str::<syn::File>(&out_string).expect("Could not parse generated bindings");

    let petsc_consts_to_get_as_regex_string = CONSTS_TO_GET_REGEX.join("|");
    let petsc_consts_to_get_as_regex = Regex::new(&petsc_consts_to_get_as_regex_string).unwrap();
    let petsc_consts = raw.items.iter()
        .filter_map(|item| match item {
            syn::Item::Const(c_item) => match c_item.expr.as_ref() {
                syn::Expr::Lit(lit) =>
                    Some((format!("{}", c_item.ident.to_token_stream()),
                        format!("{}", lit.lit.to_token_stream()))),
                _ => None,
            },
            _ => None
        })
        .filter(|(ident, _)| petsc_consts_to_get_as_regex.is_match(ident))
        .collect::<Vec<_>>();

    return PetscProber { lib, build_data: petsc_consts };
}

impl PetscProber {
    pub fn get_version_from_consts(&self) -> Version
    {
        // This code tries to emulate the code at: https://github.com/petsc/petsc/blob/e4f26ec/setup.py#L247

        // Looks at all variables named `PETSC_VERSION_*` to create version
        let (mut ver, prerel) = self.build_data.iter()
            .fold((Version::new(0,0,0), false), |(mut ver, prerel), (ident, lit)| 
                if ident == "PETSC_VERSION_MAJOR" {
                    ver.major = lit.parse().unwrap();
                    (ver, prerel)
                } else if ident == "PETSC_VERSION_MINOR" {
                    ver.minor = lit.parse().unwrap();
                    (ver, prerel)
                } else if ident == "PETSC_VERSION_SUBMINOR" {
                    ver.patch = lit.parse().unwrap();
                    (ver, prerel)
                } else if ident == "PETSC_VERSION_RELEASE" {
                    let isrel: u32 = lit.parse().unwrap();
                    (ver, isrel == 0)
                } else { 
                    (ver, prerel)
                });

        if prerel {
            ver.minor += 1;
            ver.patch = 0;
            ver.pre = Prerelease::new("dev.0").unwrap();
        }

        ver
    }

    pub fn consts_contains(&self, ident: impl ToString) -> bool {
        self.build_data.contains(&(ident.to_string(), "1".into()))
    }
}
