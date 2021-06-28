#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub const PETSC_DEFAULT_INTEGER: PetscInt = PETSC_DEFAULT;
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
