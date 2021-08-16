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

#[cfg(feature = "petsc-use-complex")]
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
#[cfg(feature = "petsc-use-complex")]
impl Into<Complex<PetscReal>> for __BindgenComplex<PetscReal> {
    fn into(self) -> Complex<PetscReal> {
        // SAFETY: `__BindgenComplex<T>` and `Complex<T>` are both
        // memory layout compatible with [T; 2]
        unsafe { std::mem::transmute(self) }
    }
}

#[cfg(feature = "petsc-use-complex")]
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

// TODO: add more trait impls. Maybe use a derive macro or something (maybe try strum_macros)
// or we would add something to the `build.rs`. Also add these trait impls for more enums/types.
#[cfg(feature = "generate-enums")] // DMBoundaryType is not "generated" but we dont want optional impls either
impl std::str::FromStr for DMBoundaryType {
    type Err = std::io::Error;
    fn from_str(input: &str) -> std::result::Result<DMBoundaryType, std::io::Error> {
        // SAFETY: `DMBoundaryTypes` is a c extern which is set to be of length 8 (with the
        // last item being NULL) in `dm.c`. We only care about the first 5. Each of these
        // entries are valid c strings.
        let dm_types_p = unsafe { DMBoundaryTypes.as_ptr() };
        let dm_types_slice =  unsafe { std::slice::from_raw_parts(dm_types_p, 5) };
        if input.to_uppercase().as_str() 
            == unsafe { std::ffi::CStr::from_ptr(dm_types_slice[DMBoundaryType::DM_BOUNDARY_NONE as usize]) }.to_str().unwrap() {
            Ok(DMBoundaryType::DM_BOUNDARY_NONE)
        } else if input.to_uppercase().as_str() 
            == unsafe { std::ffi::CStr::from_ptr(dm_types_slice[DMBoundaryType::DM_BOUNDARY_GHOSTED as usize]) }.to_str().unwrap() {
            Ok(DMBoundaryType::DM_BOUNDARY_GHOSTED)
        } else if input.to_uppercase().as_str() 
            == unsafe { std::ffi::CStr::from_ptr(dm_types_slice[DMBoundaryType::DM_BOUNDARY_MIRROR as usize]) }.to_str().unwrap() {
            Ok(DMBoundaryType::DM_BOUNDARY_MIRROR)
        } else if input.to_uppercase().as_str() 
            == unsafe { std::ffi::CStr::from_ptr(dm_types_slice[DMBoundaryType::DM_BOUNDARY_PERIODIC as usize]) }.to_str().unwrap() {
            Ok(DMBoundaryType::DM_BOUNDARY_PERIODIC)
        } else if input.to_uppercase().as_str() 
            == unsafe { std::ffi::CStr::from_ptr(dm_types_slice[DMBoundaryType::DM_BOUNDARY_TWIST as usize]) }.to_str().unwrap() {
            Ok(DMBoundaryType::DM_BOUNDARY_TWIST)
        } else {
            Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("{}, is not valid", input)))
        }
    }
}

#[cfg(feature = "generate-enums")]
impl std::fmt::Display for DMBoundaryType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // SAFETY: this c array is defined in dm.c and is defined to be of size 8.
        // But the header petscdm.h (and then bindgen) defines it as size 0 array.
        // Thus we have to trick rust into thinking it is size 5, because that's all
        // we care about.
        let dm_types_p = unsafe { DMBoundaryTypes.as_ptr() };
        let dm_types_slice =  unsafe { std::slice::from_raw_parts(dm_types_p, 5) };
        write!(f, "{}", unsafe { 
            std::ffi::CStr::from_ptr(dm_types_slice[*self as usize]) }.to_str().unwrap())
    }
}

#[cfg(feature = "generate-enums")]
impl Default for DMBoundaryType {
    fn default() -> Self { DMBoundaryType::DM_BOUNDARY_NONE }
}
