#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// Redefined some stuff from `petscsys.h`
// Im creating variables with the fortran binding names when I can
// I would like to have these defined in the petsc_wrapper.h file, but 
// bindgen seems to not be able to turn them into rust equivalents like bellow.
// https://github.com/rust-lang/rust-bindgen/issues/316
pub const PETSC_DECIDE_INTEGER: PetscInt = -1;
pub const PETSC_DETERMINE_INTEGER: PetscInt = PETSC_DECIDE_INTEGER;

pub const PETSC_DEFAULT_INTEGER: PetscInt = -2;
pub const PETSC_DEFAULT_REAL: PetscReal = -2.0;

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
