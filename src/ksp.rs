use crate::prelude::*;

// https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/index.html

pub struct KSP<'a> {
    petsc: &'a crate::Petsc,
    pub(crate) ksp_p: *mut petsc_raw::_p_KSP, // I could use KSP which is the same thing, but i think using a pointer is more clear

    // `owned_pc` is the internal pc created by the ksp object. The user code (the rust code) doesn't own it.
    // And everything (like calling destroy) is all handled internally. Calling PCDestroy our self will cause
    // problems internally. That's why we have it in a `ManuallyDrop` so PCDestroy is never called.
    // This really just serves as a reference to the PC. But we can't create a reference to a rust type if 
    // it doesn't exist anywhere. Also, the PC object does contain some members that we do want to drop, 
    // like `ref_amat` which is a `Rc` (we need to decrement the reference count) so rust can drop the
    // the Mat (by calling MatDestroy). We need to do this because the Mat was created by the user and thus
    // needs to be dropped by the user (or by rust in this case).
    // Note, if a higher level type contains a ManuallyDrop of this type, we expect that it will manually drop 
    // the members that need to be dropped, like `set_pc` and everything in `owned_pc`.
    #[doc(hidden)]
    owned_pc: Option<ManuallyDrop<PC<'a>>>,
}

impl<'a> Drop for KSP<'a> {
    fn drop(&mut self) {
        unsafe {
            let ierr = petsc_raw::KSPDestroy(&mut self.ksp_p as *mut *mut petsc_raw::_p_KSP);
            let _ = self.petsc.check_error(ierr); // TODO: should i unwrap or what idk?
        }
    }
}

impl_petsc_object_funcs!{ KSP, ksp_p }

impl<'a> KSP<'a> {
    /// Same as `KSP { ... }` but sets all optional params to `None`
    pub(crate) fn new(petsc: &'a crate::Petsc, ksp_p: *mut petsc_raw::_p_KSP) -> Self {
        KSP { petsc, ksp_p, owned_pc: None }
    }

    /// Same as [`Petsc::ksp_create()`].
    pub fn create(petsc: &'a crate::Petsc) -> Result<Self> {
        let mut ksp_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::KSPCreate(petsc.world.as_raw(), ksp_p.as_mut_ptr()) };
        petsc.check_error(ierr)?;

        Ok(KSP::new(petsc, unsafe { ksp_p.assume_init() }))
    }

    // /// Sets the matrix associated with the linear system and a (possibly)
    // /// different one associated with the preconditioner.
    // ///
    // /// Passing a `None` for `a_mat` or `p_mat` removes the matrix that is currently used.
    // pub fn set_operators(&mut self, a_mat: Option<&Mat>, p_mat: Option<&Mat>) -> Result<()>
    // {
    //     // TODO: i am 100% not doing this correctly. This is very unsafe and should changed to work better with the docs
    //     // https://petsc.org/release/docs/manualpages/KSP/KSPSetOperators.html#KSPSetOperators
    //     // TODO: if you set a_mat and p_mat to be the same, under the hood they are refrenc counted, our current setup doesn't
    //     // account for that
    //     let ierr = unsafe { petsc_raw::KSPSetOperators(self.ksp_p, a_mat.map_or(std::ptr::null_mut(), |m| m.mat_p), 
    //         p_mat.map_or(std::ptr::null_mut(), |m| m.mat_p)) };
    //     self.petsc.check_error(ierr)
    // }

    /// Sets the matrix associated with the linear system and a (possibly)
    /// different one associated with the preconditioner. Note, this is the same as
    /// `ksp.get_pc_mut().set_operators()`.
    ///
    /// Passing a `None` for `a_mat` or `p_mat` removes the matrix that is currently used.
    pub unsafe fn set_operators(&mut self, a_mat: Option<&Mat<'a>>, p_mat: Option<&Mat<'a>>) -> Result<()>
    {
        self.get_pc_mut()?.set_operators(a_mat, p_mat)
    }

    wrap_simple_petsc_member_funcs! {
        KSPSetFromOptions, set_from_options, ksp_p, #[doc = "Sets KSP options from the options database. This routine must be called before KSPSetUp() if the user is to be allowed to set the Krylov type."];
        KSPSetUp, set_up, ksp_p, #[doc = "Sets up the internal data structures for the later use of an iterative solver."];
    }

    wrap_simple_petsc_member_funcs! {
        KSPGetIterationNumber, get_iteration_number, ksp_p, i32, #[doc = "Gets the current iteration number; if the KSPSolve() is complete, returns the number of iterations used."];

    }

    /// Returns a reference to the preconditioner context set with KSPSetPC().
    pub fn get_pc<'b>(&'b mut self) -> Result<&'b PC<'a>> // IDK if this has to be a `&mut self` call
    {
        // Note, the PC object is handled by the ksp object. Thus we only want to return a reference to it
        // The KSPDestroy function will also call PCDestroy for us

        // TODO: this will create a new PC object, will this cause problems if we already have a reference out
        // Should we be able to call this method twice? do we want to use the existing pc if we can?
        // should we even have a non mut one

        // TODO: once we call `KSPGetPC` it sets the value, so should we also check if self.owned_pc is Some
        // and return self.owned_pc.as_ref().unwrap() if so. Note calling `KSPGetPC` should be the same, but
        // with the added cost of calling the function. I dont see why we can't do this.
        let mut pc_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::KSPGetPC(self.ksp_p, pc_p.as_mut_ptr()) };
        self.petsc.check_error(ierr)?;

        self.owned_pc = Some(ManuallyDrop::new(PC::new(&self.petsc, unsafe { pc_p.assume_init() })));
        
        Ok(self.owned_pc.as_ref().unwrap())
    }

    /// Returns a mutable reference to the preconditioner context set with KSPSetPC().
    pub fn get_pc_mut<'b>(&'b mut self) -> Result<&'b mut PC<'a>> // IDK if this has to be a `&mut self` call
    {
        // Note, the PC object is handled by the ksp object. Thus we only want to return a reference to it
        // The KSPDestroy function will also call PCDestroy for us

        // TODO: this will create a new PC object, will this cause problems if we already have a reference out
        // Should we be able to call this method twice

        let mut pc_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::KSPGetPC(self.ksp_p, pc_p.as_mut_ptr()) };
        self.petsc.check_error(ierr)?;

        self.owned_pc = Some(ManuallyDrop::new(PC::new(&self.petsc, unsafe { pc_p.assume_init() })));
        
        Ok(self.owned_pc.as_mut().unwrap())
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
        self.petsc.check_error(ierr)
    }

    /// Solves linear system.
    pub fn solve(&mut self, b: &Vector, x: &mut Vector) -> Result<()>
    {
        let ierr = unsafe { petsc_raw::KSPSolve(self.ksp_p, b.vec_p, x.vec_p) };
        self.petsc.check_error(ierr)
    }

}
