//! Raw PETSc bindings generated by bindgen.
//!
//! # Features
//! 
//! PETSc has support for multiple different sizes of scalars and integers. To expose this
//! to rust, we require you set different features. The following are all the features that
//! can be set. Note, you are required to have exactly one scalar feature set and exactly
//! one integer feature set. And it must match the PETSc install.
//!
//! - **`petsc-real-f64`** *(enabled by default)* — Sets the real type, [`PetscReal`], to be `f64`.
//! Also sets the complex type, [`PetscComplex`], to be `Complex<f64>`.
//! - **`petsc-real-f32`** — Sets the real type, [`PetscReal`] to be `f32`.
//! Also sets the complex type, [`PetscComplex`], to be `Complex<f32>`.
//! - **`petsc-use-complex`** *(disabled by default)* *(experimental only)* - Sets the scalar type, [`PetscScalar`], to
//! be the complex type, [`PetscComplex`]. If disabled then the scalar type is the real type, [`PetscReal`].
//! You must be using the `complex-scalar` branch to enable this feature.
//! - **`petsc-int-i32`** *(enabled by default)* — Sets the integer type, [`PetscInt`], to be `i32`.
//! - **`petsc-int-i64`** — Sets the integer type, [`PetscInt`], to be `i64`.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(feature = "petsc-use-complex")]
use num_complex::Complex;

pub const PETSC_DECIDE_INTEGER: PetscInt = PETSC_DECIDE as PetscInt;
pub const PETSC_DEFAULT_INTEGER: PetscInt = PETSC_DEFAULT as PetscInt;
pub const PETSC_DEFAULT_REAL: PetscReal = PETSC_DEFAULT as PetscReal;

// Redefined stuff from `petscpctypes.h`
/// This table is from: <https://petsc.org/release/docs/manualpages/PC/PCType.html#PCType>
pub static PCTYPE_TABLE: &'static [&str] = &[
    "none",
    "jacobi",
    "sor",
    "lu",
    "shell",
    "bjacobi",
    "mg",
    "eisenstat",
    "ilu",
    "icc",
    "asm",
    "gasm",
    "ksp",
    "composite",
    "redundant",
    "spai",
    "nn",
    "cholesky",
    "pbjacobi",
    "vpbjacobi",
    "mat",
    "hypre",
    "parms",
    "fieldsplit",
    "tfs",
    "ml",
    "galerkin",
    "exotic",
    "cp",
    "bfbt",
    "lsc",
    "python",
    "pfmg",
    "syspfmg",
    "redistribute",
    "svd",
    "gamg",
    "chowiluviennacl",
    "rowscalingviennacl",
    "saviennacl",
    "bddc",
    "kaczmarz",
    "telescope",
    "patch",
    "lmvm",
    "hmg",
    "deflation",
    "hpddm",
    "hara"
];

/// Preconditioner type.
///
/// This enum is from: <https://petsc.org/release/docs/manualpages/PC/PCType.html#PCType>
pub enum PCTypeEnum {
    PCNONE = 0,
    PCJACOBI,
    PCSOR,
    PCLU,
    PCSHELL,
    PCBJACOBI,
    PCMG,
    PCEISENSTAT,
    PCILU,
    PCICC,
    PCASM,
    PCGASM,
    PCKSP,
    PCCOMPOSITE,
    PCREDUNDANT,
    PCSPAI,
    PCNN,
    PCCHOLESKY,
    PCPBJACOBI,
    PCVPBJACOBI,
    PCMAT,
    PCHYPRE,
    PCPARMS,
    PCFIELDSPLIT,
    PCTFS,
    PCML,
    PCGALERKIN,
    PCEXOTIC,
    PCCP,
    PCBFBT,
    PCLSC,
    PCPYTHON,
    PCPFMG,
    PCSYSPFMG,
    PCREDISTRIBUTE,
    PCSVD,
    PCGAMG,
    PCCHOWILUVIENNACL,
    PCROWSCALINGVIENNACL,
    PCSAVIENNACL,
    PCBDDC,
    PCKACZMARZ,
    PCTELESCOPE,
    PCPATCH,
    PCLMVM,
    PCHMG,
    PCDEFLATION,
    PCHPDDM,
    PCHARA,
}

// Redefined stuff from `petscpctypes.h`
/// This table is from: <https://petsc.org/release/docs/manualpages/PC/PCType.html#PCType>
pub static DMTYPE_TABLE: &'static [&str] = &[
    "da",
    "composite",
    "sliced",
    "shell",
    "plex",
    "redundant",
    "patch",
    "moab",
    "network",
    "forest",
    "p4est",
    "p8est",
    "swarm",
    "product",
    "stag",
];

/// This enum is from: <https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DM/DMType.html#DMType>
pub enum DMTypeEnum {
    DMDA = 0,
    DMCOMPOSITE,
    DMSLICED,
    DMSHELL,
    DMPLEX,
    DMREDUNDANT,
    DMPATCH,
    DMMOAB,
    DMNETWORK,
    DMFOREST,
    DMP4EST,
    DMP8EST,
    DMSWARM,
    DMPRODUCT,
    DMSTAG,
}

// TODO: num_complex::Complex docs say:
// "Note that Complex<F> where F is a floating point type is only memory layout compatible with C’s
// complex types, not necessarily calling convention compatible. This means that for FFI you can
// only pass Complex<F> behind a pointer, not as a value."
// What does this mean for use? If every function is listed as `extern "C"` are we ok?
// Petsc function dont take pointers as inputs all the time.
#[cfg(feature = "petsc-use-complex")]
impl Into<Complex<PetscReal>> for __BindgenComplex<PetscReal> {
    fn into(self) -> Complex<PetscReal> {
        // This should be safe because `__BindgenComplex<T>` and `Complex<T>` are both
        // memory layout compatible with an array [T; 2]
        unsafe { std::mem::transmute(self) }
    }
}

#[cfg(feature = "petsc-use-complex")]
impl From<Complex<PetscReal>> for __BindgenComplex<PetscReal> {
    fn from(ct: Complex<PetscReal>) -> __BindgenComplex<PetscReal> {
        // This should be safe because `__BindgenComplex` and `Complex` are both
        // memory layout compatible with an array [T; 2]
        unsafe { std::mem::transmute(ct) }
    }
}

impl Into<bool> for PetscBool {
    fn into(self) -> bool {
        match self {
            PetscBool::PETSC_FALSE => false,
            PetscBool::PETSC_TRUE => true
        }
    }
}

impl From<bool> for PetscBool {
    fn from(b: bool) -> Self {
        match b {
            false => PetscBool::PETSC_FALSE,
            true => PetscBool::PETSC_TRUE
        }
    }
}
