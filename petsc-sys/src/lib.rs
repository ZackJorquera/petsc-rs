#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]
#![allow(non_snake_case)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// redefined some stuff from `petscsys.h`
// Im creating variables with the fortran binding names when I can
// I would like to have these defined in the petsc_wrapper.h file, but 
// bindgen seems to not be able to turn them into rust equivalents like bellow.
pub const PETSC_DECIDE_INTEGER: PetscInt = -1;
pub const PETSC_DETERMINE_INTEGER: PetscInt = PETSC_DECIDE_INTEGER;

pub const PETSC_DEFAULT_INTEGER: PetscInt = -2;
pub const PETSC_DEFAULT_REAL: PetscReal = -2.0;
