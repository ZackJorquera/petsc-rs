use semver::{Prerelease, VersionReq};

fn main() {
    let header_version = build_probe_petsc::probe("3.15").get_version_from_headers();

    // we assume that there is only one version of pre-release,
    // which is currently the case for PETSc.
    let petsc_version_prerelease = !header_version.pre.is_empty();
    let header_version_without_pre = {
        let mut hv = header_version.clone();
        hv.pre = Prerelease::EMPTY;
        hv
    };
    
    // We remove the pre-release because it makes things hard to compare
    let petsc_version_3_16 = VersionReq::parse("=3.16").unwrap().matches(&header_version_without_pre);
    let petsc_version_ge_3_16 = VersionReq::parse(">=3.16.0").unwrap().matches(&header_version_without_pre);
    let petsc_version_3_15 = VersionReq::parse("~3.15").unwrap().matches(&header_version_without_pre);
    let petsc_version_3 = VersionReq::parse("^3.15").unwrap().matches(&header_version_without_pre);
    let petsc_version_ge_4 = VersionReq::parse(">=4").unwrap().matches(&header_version_without_pre);

    if petsc_version_3_16 && petsc_version_prerelease {
        println!("cargo:warning={}", "The current PETSc build is pre-release. petsc-rs will have unstable features");
        println!("cargo:rustc-cfg=petsc_version_3_16_dev");
    } else if petsc_version_ge_3_16 {
        println!("cargo:rustc-cfg=petsc_version_ge_3_16");
        panic!("petsc-rs doesn't currently support PETSc v3.16 or above.")
    } else if petsc_version_3_15 {
        println!("cargo:rustc-cfg=petsc_version_3_15");
    }

    if petsc_version_3 {
        println!("cargo:rustc-cfg=petsc_version_3");
    } else if petsc_version_ge_4 {
        println!("cargo:rustc-cfg=petsc_version_ge_4");
        panic!("petsc-rs doesn't currently support PETSc v4 or above.")
    }
}
