//! The Scalable Linear Equations Solvers (KSP) component provides an easy-to-use interface to 
//! the combination of a Krylov subspace iterative method and a preconditioner (in the [KSP](crate::ksp) and [PC](crate::pc) 
//! components, respectively) or a sequential direct solver. 
//!
//! KSP users can set various preconditioning options at runtime via the options database 
//! (e.g., -pc_type jacobi ). KSP users can also set PC options directly in application codes by 
//! first extracting the PC context from the KSP context via [`KSP::get_pc()`] and then directly
//! calling the PC routines listed below (e.g., [`PC::set_type()`]). PC components can be used directly
//! to create and destroy solvers; this is not needed for users but is for library developers.
//!
//! PETSc C API docs: <https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/PC/index.html>

use crate::prelude::*;

pub use crate::petsc_raw::PCTypeEnum as PCType;

/// Abstract PETSc object that manages all preconditioners including direct solvers such as PCLU
pub struct PC<'a> {
    pub(crate) world: &'a dyn Communicator,
    pub(crate) pc_p: *mut petsc_raw::_p_PC, // I could use petsc_raw::PC which is the same thing, but i think using a pointer is more clear

    // We take an `Rc` because we don't want ownership of the Mat. Under the hood, this is how the 
    // PetscSetOperators function works, it increments the reference count. The problem with this
    // solution right now is that we lose mutable access. It might be worth making it a Rc<RefCell<Mat>>.
    // This might also allow us to have a get_operators function (which would also returns a Rc<RefCell<Mat>>).
    // Regardless, returning mutable access would be hard, especially when the rust side can't guarantee how the 
    // C api accesses the operators behind the scenes.
    ref_amat: Option<Rc<Mat<'a>>>,
    ref_pmat: Option<Rc<Mat<'a>>>,
}

impl<'a> Drop for PC<'a> {
    // Note, this should only be called if the PC context was created with `PCCreate`.
    fn drop(&mut self) {
        unsafe {
            let ierr = petsc_raw::PCDestroy(&mut self.pc_p as *mut *mut petsc_raw::_p_PC);
            let _ = Petsc::check_error(self.world, ierr); // TODO: should i unwrap or what idk?
        }
    }
}

impl<'a> PC<'a> {
    /// Same as `PC { ... }` but sets all optional params to `None`
    pub(crate) fn new(world: &'a dyn Communicator, pc_p: *mut petsc_raw::_p_PC) -> Self {
        PC { world, pc_p, ref_amat: None, ref_pmat: None }
    }

    /// Creates a preconditioner context.
    ///
    /// You will most likely create a preconditioner context from a solver type such as
    /// from a Krylov solver, [`KSP`], using the [`KSP::get_pc()`] method.
    ///
    /// [`KSP::get_pc`]: KSP::get_pc
    pub fn create(world: &'a dyn Communicator) -> Result<Self> {
        let mut pc_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::PCCreate(world.as_raw(), pc_p.as_mut_ptr()) };
        Petsc::check_error(world, ierr)?;

        Ok(PC::new(world, unsafe { pc_p.assume_init() }))
    }

    /// Builds PC for a particular preconditioner type
    pub fn set_type(&mut self, pc_type: PCType) -> Result<()>
    {
        let option_str = petsc_raw::PCTYPE_TABLE[pc_type as usize];
        let cstring = CString::new(option_str).expect("`CString::new` failed");
        let ierr = unsafe { petsc_raw::PCSetType(self.pc_p, cstring.as_ptr()) };
        Petsc::check_error(self.world, ierr)
    }
    
    /// Sets the matrix associated with the linear system and a (possibly)
    /// different one associated with the preconditioner.
    ///
    /// Passing a `None` for `a_mat` or `p_mat` removes the matrix that is currently used.
    // TODO: should we pass in `Rc`s or should we just transfer ownership.
    // or we could do `Rc<RefCell<Mat>>` so that when you remove the mats we can give mut access back.
    pub fn set_operators(&mut self, a_mat: Option<Rc<Mat<'a>>>, p_mat: Option<Rc<Mat<'a>>>) -> Result<()>
    {
        // Should this function consume the mats? Right now once call this function you can not edit the mats with
        // out first removing them.

        // TODO: i am 100% not doing this correctly. This is might be unsafe, here are the docs:
        // https://petsc.org/release/docs/manualpages/KSP/KSPSetOperators.html#KSPSetOperators
        // TODO: if you set a_mat and p_mat to be the same, under the hood they are refrenc counted. Does our current setup
        // account for that? idk, i think it does, but idk
        let ierr = unsafe { petsc_raw::PCSetOperators(self.pc_p,
            a_mat.as_ref().map_or(std::ptr::null_mut(), |m| m.mat_p), 
            p_mat.as_ref().map_or(std::ptr::null_mut(), |m| m.mat_p)) };
        Petsc::check_error(self.world, ierr)?;

        // drop everything as it is getting replaced. (note under the hood MatDestroy is called on both of them).
        // let _ = self.owned_amat.take();
        // let _ = self.owned_pmat.take();
        let _ = self.ref_amat.take();
        let _ = self.ref_pmat.take();

        self.ref_amat = a_mat;
        self.ref_pmat = p_mat;

        Ok(())
    }
}

// Macro impls
impl<'a> PC<'a> {
    wrap_simple_petsc_member_funcs! {
        PCSetFromOptions, set_from_options, pc_p, #[doc = "Sets PC options from the options database. This routine must be called before PCSetUp() if the user is to be allowed to set the preconditioner method."];
        PCSetUp, set_up, pc_p, #[doc = "Prepares for the use of a preconditioner."];
    }
}

impl_petsc_object_funcs!{ PC, pc_p }
