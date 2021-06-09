//! The Scalable Linear Equations Solvers (KSP) component provides an easy-to-use interface to 
//! the combination of a Krylov subspace iterative method and a preconditioner (in the [KSP](ksp) and [PC](pc) 
//! components, respectively) or a sequential direct solver. 
//!
//! KSP users can set various preconditioning options at runtime via the options database 
//! (e.g., -pc_type jacobi ). KSP users can also set PC options directly in application codes by 
//! first extracting the PC context from the KSP context via [`KSP::get_pc()`] and then directly
//! calling the PC routines listed below (e.g., [`PC::set_type()`]). PC components can be used directly
//! to create and destroy solvers; this is not needed for users but is for library developers.

use crate::prelude::*;

// https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/PC/index.html

// This table is from: https://petsc.org/release/docs/manualpages/PC/PCType.html#PCType
static PCTYPE_TABLE: &'static [&str] = &[
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

pub enum PCType {
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

/// Abstract PETSc object that manages all preconditioners including direct solvers such as PCLU
pub struct PC<'a> {
    pub(crate) petsc: &'a crate::Petsc,
    pub(crate) pc_p: *mut petsc_raw::_p_PC, // I could use PC which is the same thing, but i think using a pointer is more clear
}

impl<'a> Drop for PC<'a> {
    // Note, this should only be called if the PC context was created with `PCCreate`.
    fn drop(&mut self) {
        unsafe {
            let ierr = petsc_raw::PCDestroy(&mut self.pc_p as *mut *mut petsc_raw::_p_PC);
            let _ = self.petsc.check_error(ierr); // TODO: should i unwrap or what idk?

            println!("drop called for PC");
        }
    }
}

impl_petsc_object_funcs!{ PC, pc_p }

impl<'a> PC<'a> {
    /// Same as `PC { ... }` but sets all optional params to `None`
    pub(crate) fn new(petsc: &'a crate::Petsc, pc_p: *mut petsc_raw::_p_PC) -> Self {
        PC { petsc, pc_p }
    }

    /// Creates a preconditioner context.
    ///
    /// You will most likely create a preconditioner context from a solver type such as
    /// from a Krylov solver, [`KSP`], using the [`KSP::get_pc()`] method.
    ///
    /// [`KSP::get_pc`]: KSP::get_pc
    pub fn create(petsc: &'a crate::Petsc) -> Result<Self> {
        let mut pc_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::PCCreate(petsc.world.as_raw(), pc_p.as_mut_ptr()) };
        petsc.check_error(ierr)?;

        Ok(PC::new(petsc, unsafe { pc_p.assume_init() }))
    }

    wrap_simple_petsc_member_funcs! {
        PCSetFromOptions, set_from_options, pc_p, #[doc = "Sets PC options from the options database. This routine must be called before PCSetUp() if the user is to be allowed to set the preconditioner method."];
        PCSetUp, set_up, pc_p, #[doc = "Prepares for the use of a preconditioner."];
    }

    /// Builds PC for a particular preconditioner type
    pub fn set_type(&mut self, pc_type: PCType) -> Result<()>
    {
        let option_str = PCTYPE_TABLE[pc_type as usize];
        let cstring = CString::new(option_str).expect("`CString::new` failed");
        let ierr = unsafe { petsc_raw::PCSetType(self.pc_p, cstring.as_ptr()) };
        self.petsc.check_error(ierr)
    }

    /// Sets the matrix associated with the linear system and a (possibly)
    /// different one associated with the preconditioner.
    ///
    /// Passing a `None` for `a_mat` or `p_mat` removes the matrix that is currently used.
    pub unsafe fn set_operators(&mut self, a_mat: Option<&Mat<'a>>, p_mat: Option<&Mat<'a>>) -> Result<()>
    {
        // Should this function consume the mats? Right now once call this function you can not edit the mats with
        // out first removing them.

        // TODO: i am 100% not doing this correctly. This is unsafe, here are the docs:
        // https://petsc.org/release/docs/manualpages/KSP/KSPSetOperators.html#KSPSetOperators
        // TODO: if you set a_mat and p_mat to be the same, under the hood they are refrenc counted. Our current setup
        // does not account for that. also pointers to the a_mat and p_mat are stored in the c struct so 
        // We need to tell rust that this struct owns a refrence to a_mat and p_mat
        let ierr = petsc_raw::PCSetOperators(self.pc_p,
            a_mat.as_ref().map_or(std::ptr::null_mut(), |m| m.mat_p), 
            p_mat.as_ref().map_or(std::ptr::null_mut(), |m| m.mat_p));
        self.petsc.check_error(ierr)
    }
}
