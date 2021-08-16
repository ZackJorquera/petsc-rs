use std::env;
use std::path::Path;

use regex::Regex;
use semver::{Prerelease, Version};

pub struct PetscProber {
    pub lib: pkg_config::Library,
    pub petscversion_data: String,
    pub petscconf_data: String,
}

/// Wrapper around [`pkg_config::Config::probe()`].
///
/// Use [`PetscProber.lib`] to get data from [`pkg_config`].
///
/// Also read in the headers `petscversion.h` and `petscconf.h`.
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

    let mut petscversion_h = None;
    let mut petscconf_h = None;

    for dir in &lib.include_paths {
        println!("cargo:rerun-if-changed={}", dir.to_string_lossy());

        if petscversion_h.is_none() {
            let header_path = dir.join("petscversion.h");
            if header_path.exists() { petscversion_h = Some(header_path) } 
        }
        if petscconf_h.is_none() {
            let header_path = dir.join("petscconf.h");
            if header_path.exists() { petscconf_h = Some(header_path) } 
        }
    }

    let petscversion_h = petscversion_h.expect("Could not find `petscversion.h`.");
    let petscconf_h = petscconf_h.expect("Could not find `petscconf.h`.");

    // These are both small files so it doesn't matter that much, but maybe we
    // should wrap them with `Lazy` so they are are read only if they are used.
    // Note, `std::lazy::Lazy` is not stable yet.
    let petscversion_data = std::fs::read_to_string(petscversion_h).unwrap();
    let petscconf_data = std::fs::read_to_string(petscconf_h).unwrap();

    return PetscProber { lib, petscversion_data, petscconf_data };
}

impl PetscProber {
    /// Gets the [`Version`] of  PETSc from the header files. This might be different from the version
    /// in [`PetscProber.lib.version`]
    pub fn get_version_from_headers(&self) -> Version {
        // This code tries to emulate the code at: https://gitlab.com/petsc/petsc/blob/e4f26ec/setup.py#L247-L265
        let major = Regex::new(r"#define\s+PETSC_VERSION_MAJOR\s+(\d+)").unwrap();
        let minor = Regex::new(r"#define\s+PETSC_VERSION_MINOR\s+(\d+)").unwrap();
        let subminor = Regex::new(r"#define\s+PETSC_VERSION_SUBMINOR\s+(\d+)").unwrap();
        let release = Regex::new(r"#define\s+PETSC_VERSION_RELEASE\s+([-]*\d+)").unwrap();

        // We dont need to do a fold here, but it looks cool :)
        let (mut ver, prerel) = self.petscversion_data.lines()
            .fold((Version::new(0,0,0), false), |(mut ver, prerel), line| 
                if let Some(caps) =  major.captures(line) {
                    ver.major = caps.get(1).unwrap().as_str().parse().unwrap();
                    (ver, prerel)
                } else if let Some(caps) =  minor.captures(line) {
                    ver.minor = caps.get(1).unwrap().as_str().parse().unwrap();
                    (ver, prerel)
                } else if let Some(caps) =  subminor.captures(line) {
                    ver.patch = caps.get(1).unwrap().as_str().parse().unwrap();
                    (ver, prerel)
                } else if let Some(caps) =  release.captures(line) {
                    let isrel: u32 = caps.get(1).unwrap().as_str().parse().unwrap();
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

    /// Checks to see if a `#define <ident> 1` exists the headers.
    pub fn defines_contains(&self, ident: impl ToString) -> bool {
        let re_match = Regex::new(&format!("#define\\s+{}\\s+1", ident.to_string())).unwrap();

        self.petscconf_data.lines().any(|line| re_match.is_match(line))
            || self.petscversion_data.lines().any(|line| re_match.is_match(line))
    }
}
