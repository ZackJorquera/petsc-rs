#![doc = include_str!("../README.md")]

// Note to developer, to run all the tests for the generated code use:
// `cargo test --features generate-enums,use-private-headers`

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deref_nullptr)] // this is done in bindgen tests

use mpi_sys::*;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
#[cfg(feature = "generate-enums")]
include!(concat!(env!("OUT_DIR"), "/enums.rs"));

#[cfg(feature = "petsc-use-complex-unsafe")]
use num_complex::Complex;

pub const PETSC_DECIDE_INTEGER: PetscInt = PETSC_DECIDE as PetscInt;
pub const PETSC_DEFAULT_INTEGER: PetscInt = PETSC_DEFAULT as PetscInt;
pub const PETSC_DEFAULT_REAL: PetscReal = PETSC_DEFAULT as PetscReal;

// TODO: num_complex::Complex docs say:
// "Note that Complex<F> where F is a floating point type is only memory layout compatible with C’s
// complex types, not necessarily calling convention compatible. This means that for FFI you can
// only pass Complex<F> behind a pointer, not as a value."
// What does this mean for use? If every function is listed as `extern "C"` are we ok?
// Petsc function dont take pointers as inputs all the time.
#[cfg(feature = "petsc-use-complex-unsafe")]
impl Into<Complex<PetscReal>> for __BindgenComplex<PetscReal> {
    fn into(self) -> Complex<PetscReal> {
        // SAFETY: `__BindgenComplex<T>` and `Complex<T>` are both
        // memory layout compatible with [T; 2]
        unsafe { std::mem::transmute(self) }
    }
}

#[cfg(feature = "petsc-use-complex-unsafe")]
impl From<Complex<PetscReal>> for __BindgenComplex<PetscReal> {
    fn from(ct: Complex<PetscReal>) -> __BindgenComplex<PetscReal> {
        // SAFETY: `__BindgenComplex<T>` and `Complex<T>` are both
        // memory layout compatible with [T; 2]
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
