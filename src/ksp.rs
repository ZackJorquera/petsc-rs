//! The scalable linear equations solvers (KSP) component provides an easy-to-use interface to the 
//! combination of a Krylov subspace iterative method and a preconditioner (in the [KSP](crate::ksp) and [PC](crate::pc)
//! components, respectively) or a sequential direct solver. 
//! 
//! KSP users can set various Krylov subspace options at runtime via the options database 
//! (e.g., -ksp_type cg ). KSP users can also set KSP options directly in application by directly calling
//! the KSP routines listed below (e.g., [`KSP::set_type()`](#) ). KSP components can be used directly to
//! create and destroy solvers; this is not needed for users but is intended for library developers.
//!
//! PETSc C API docs: <https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/index.html>

use crate::prelude::*;

/// Abstract PETSc object that manages all Krylov methods. This is the object that manages the linear
/// solves in PETSc (even those such as direct solvers that do no use Krylov accelerators).
pub struct KSP<'a> {
    world: &'a dyn Communicator,
    pub(crate) ksp_p: *mut petsc_raw::_p_KSP, // I could use KSP which is the same thing, but i think using a pointer is more clear

    // As far as Petsc is concerned we own a reference to the PC as it is reference counted under the hood.
    // But it should be fine to keep it here as an owned reference because we can control access and the 
    // default `KSPDestroy` accounts for references.
    #[doc(hidden)]
    pc: Option<PC<'a>>
    
}

impl<'a> Drop for KSP<'a> {
    fn drop(&mut self) {
        unsafe {
            let ierr = petsc_raw::KSPDestroy(&mut self.ksp_p as *mut *mut petsc_raw::_p_KSP);
            let _ = Petsc::check_error(self.world, ierr); // TODO: should I unwrap or what idk?
        }
    }
}

impl<'a> KSP<'a> {
    /// Same as `KSP { ... }` but sets all optional params to `None`
    pub(crate) fn new(world: &'a dyn Communicator, ksp_p: *mut petsc_raw::_p_KSP) -> Self {
        KSP { world, ksp_p, pc: None }
    }

    /// Same as [`Petsc::ksp_create()`].
    pub fn create(world: &'a dyn Communicator) -> Result<Self> {
        let mut ksp_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::KSPCreate(world.as_raw(), ksp_p.as_mut_ptr()) };
        Petsc::check_error(world, ierr)?;

        Ok(KSP::new(world, unsafe { ksp_p.assume_init() }))
    }

    /// Sets the matrix associated with the linear system and a (possibly)
    /// different one associated with the preconditioner. Note, this is the same as
    /// `ksp.get_pc_mut().set_operators()`.
    ///
    /// Passing a `None` for `a_mat` or `p_mat` removes the matrix that is currently used.
    pub fn set_operators(&mut self, a_mat: Option<Rc<Mat<'a>>>, p_mat: Option<Rc<Mat<'a>>>) -> Result<()>
    {
        // TODO: should we call `KSPSetOperators`? or should we just call `PC::set_operators`
        // The source for KSPSetOperators basically just calls PCSetOperators but does something with 
        // `ksp->setupstage` so idk.
        self.get_pc_mut()?.set_operators(a_mat, p_mat)
    }

    /// Sets the preconditioner to be used to calculate the application of the preconditioner on a vector.
    /// if you change the PC by calling set again, then the original will be dropped.
    pub fn set_pc(&mut self, pc: PC<'a>) -> Result<()>
    {
        
        let ierr = unsafe { petsc_raw::KSPSetPC(self.ksp_p, pc.pc_p) };
        Petsc::check_error(self.world, ierr)?;

        let _ = self.pc.take();
        self.pc = Some(pc);

        Ok(())
    }

    /// Returns a reference to the preconditioner context set with KSPSetPC().
    pub fn get_pc<'b>(&'b mut self) -> Result<&'b PC<'a>> // IDK if this has to be a `&mut self` call
    {
        // TODO: should we even have a non mut one (or only `get_pc_mut`)

        // Under the hood, if the pc is already set, i.e. with `set_pc`, then `KSPGetPC` just returns a pointer
        // to that, so we can bypass calling KSPGetPC. However, there shouldn't be any problem with just calling
        // `KSPGetPC` again as we incremented the reference of the PC we "own" so dropping it wont do anything.
        if let Some(ref pc) = self.pc
        {
            Ok(pc)
        }
        else
        {
            let mut pc_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::KSPGetPC(self.ksp_p, pc_p.as_mut_ptr()) };
            Petsc::check_error(self.world, ierr)?;
            let ierr = unsafe { petsc_raw::PetscObjectReference(pc_p.assume_init() as *mut petsc_raw::_p_PetscObject) };
            Petsc::check_error(self.world, ierr)?;

            self.pc = Some(PC::new(self.world, unsafe { pc_p.assume_init() }));

            Ok(self.pc.as_ref().unwrap())
        }
    }

    /// Returns a mutable reference to the preconditioner context set with KSPSetPC().
    pub fn get_pc_mut<'b>(&'b mut self) -> Result<&'b mut PC<'a>> // IDK if this has to be a `&mut self` call
    {
        if let Some(ref mut pc) = self.pc
        {
            Ok(pc)
        }
        else
        {
            let mut pc_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::KSPGetPC(self.ksp_p, pc_p.as_mut_ptr()) };
            Petsc::check_error(self.world, ierr)?;
            let ierr = unsafe { petsc_raw::PetscObjectReference(pc_p.assume_init() as *mut petsc_raw::_p_PetscObject) };
            Petsc::check_error(self.world, ierr)?;

            self.pc = Some(PC::new(self.world, unsafe { pc_p.assume_init() }));

            Ok(self.pc.as_mut().unwrap())
        }
    }

    /// Sets the relative, absolute, divergence, and maximum iteration tolerances 
    /// used by the default KSP convergence testers.
    ///
    /// Set the inputs to be `None` If you wish to use the default value of any of the tolerances.
    ///
    /// Parameters.
    /// 
    /// * `rtol` - The relative convergence tolerance, relative decrease in the (possibly preconditioned) residual norm
    /// * `atol` - The absolute convergence tolerance absolute size of the (possibly preconditioned) residual norm
    /// * `dtol` - the divergence tolerance, amount (possibly preconditioned) residual norm can increase before KSPConvergedDefault() concludes that the method is diverging
    /// * `max_iters` - Maximum number of iterations to use
    pub fn set_tolerances(&mut self, rtol: Option<f64>, atol: Option<f64>, 
            dtol: Option<f64>, max_iters: Option<i32>) -> Result<()>
    {
        let ierr = unsafe { petsc_raw::KSPSetTolerances(
            self.ksp_p, rtol.unwrap_or(petsc_raw::PETSC_DEFAULT_REAL), atol.unwrap_or(petsc_raw::PETSC_DEFAULT_REAL),
            dtol.unwrap_or(petsc_raw::PETSC_DEFAULT_REAL), max_iters.unwrap_or(petsc_raw::PETSC_DEFAULT_INTEGER)) };
        Petsc::check_error(self.world, ierr)
    }

    /// Solves linear system.
    pub fn solve(&self, b: &Vector, x: &mut Vector) -> Result<()>
    {
        let ierr = unsafe { petsc_raw::KSPSolve(self.ksp_p, b.vec_p, x.vec_p) };
        Petsc::check_error(self.world, ierr)
    }

}

// macro impls
impl<'a> KSP<'a> {
    wrap_simple_petsc_member_funcs! {
        KSPSetFromOptions, set_from_options, ksp_p, takes mut, #[doc = "Sets KSP options from the options database. This routine must be called before KSPSetUp() if the user is to be allowed to set the Krylov type."];
        KSPSetUp, set_up, ksp_p, takes mut, #[doc = "Sets up the internal data structures for the later use of an iterative solver. . This will be automatically called with [`KSP::solve()`]."];
        KSPGetIterationNumber, get_iteration_number, ksp_p, output i32, iter_num, #[doc = "Gets the current iteration number; if the KSPSolve() is complete, returns the number of iterations used."];
    }
}

impl_petsc_object_funcs!{ KSP, ksp_p }

impl_petsc_view_func!{ KSP, ksp_p, KSPView }
