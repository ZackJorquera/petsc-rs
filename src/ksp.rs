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

use std::{mem::ManuallyDrop, pin::Pin};

use crate::prelude::*;

/// Abstract PETSc object that manages all Krylov methods. This is the object that manages the linear
/// solves in PETSc (even those such as direct solvers that do no use Krylov accelerators).
pub struct KSP<'a, 'tl> {
    world: &'a dyn Communicator,
    pub(crate) ksp_p: *mut petsc_raw::_p_KSP, // I could use KSP which is the same thing, but i think using a pointer is more clear

    // As far as Petsc is concerned we own a reference to the PC as it is reference counted under the hood.
    // But it should be fine to keep it here as an owned reference because we can control access and the 
    // default `KSPDestroy` accounts for references.
    pc: Option<PC<'a>>,

    dm: Option<DM<'a>>,

    compute_operators_trampoline_data: Option<Pin<Box<KSPComputeOperatorsTrampolineData<'a, 'tl>>>>,
    compute_rhs_trampoline_data: Option<Pin<Box<KSPComputeRHSTrampolineData<'a, 'tl>>>>,
}

struct KSPComputeOperatorsTrampolineData<'a, 'tl> {
    world: &'a dyn Communicator,
    user_f: Box<dyn FnMut(&KSP<'a, 'tl>, &DM<'a>, &mut Mat<'a>, &mut Mat<'a>) -> Result<()> + 'tl>,
}

struct KSPComputeRHSTrampolineData<'a, 'tl> {
    world: &'a dyn Communicator,
    user_f: Box<dyn FnMut(&KSP<'a, 'tl>, &DM<'a>, &mut Vector<'a>) -> Result<()> + 'tl>,
}

impl<'a> Drop for KSP<'a, '_> {
    fn drop(&mut self) {
        unsafe {
            let ierr = petsc_raw::KSPDestroy(&mut self.ksp_p as *mut *mut petsc_raw::_p_KSP);
            let _ = Petsc::check_error(self.world, ierr); // TODO: should I unwrap or what idk?
        }
    }
}

impl<'a, 'tl> KSP<'a, 'tl> {
    /// Same as `KSP { ... }` but sets all optional params to `None`
    pub(crate) fn new(world: &'a dyn Communicator, ksp_p: *mut petsc_raw::_p_KSP) -> Self {
        KSP { world, ksp_p, pc: None, dm: None,
            compute_operators_trampoline_data: None,
            compute_rhs_trampoline_data: None }
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

    /// Sets the [preconditioner](crate::pc) to be used to calculate the application of
    /// the preconditioner on a vector.
    ///
    /// if you change the PC by calling set again, then the original will be dropped.
    pub fn set_pc(&mut self, pc: PC<'a>) -> Result<()>
    {
        
        let ierr = unsafe { petsc_raw::KSPSetPC(self.ksp_p, pc.pc_p) };
        Petsc::check_error(self.world, ierr)?;

        let _ = self.pc.take();
        self.pc = Some(pc);

        Ok(())
    }

    /// Returns a reference to the [preconditioner](crate::pc) context set with [`KSP::set_pc()`].
    pub fn get_pc<'b>(&'b mut self) -> Result<&'b PC<'a>>
    {
        // TODO: should we even have a non mut one (or only `get_pc_mut`)

        // Under the hood, if the pc is already set, i.e. with `set_pc`, then `KSPGetPC` just returns a pointer
        // to that, so we can bypass calling KSPGetPC. However, there shouldn't be any problem with just calling
        // `KSPGetPC` again as we incremented the reference of the PC we "own" so dropping it wont do anything.
        if self.pc.is_some() {
            Ok(self.pc.as_ref().unwrap())
        } else {
            let mut pc_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::KSPGetPC(self.ksp_p, pc_p.as_mut_ptr()) };
            Petsc::check_error(self.world, ierr)?;
            let ierr = unsafe { petsc_raw::PetscObjectReference(pc_p.assume_init() as *mut petsc_raw::_p_PetscObject) };
            Petsc::check_error(self.world, ierr)?;

            self.pc = Some(PC::new(self.world, unsafe { pc_p.assume_init() }));

            Ok(self.pc.as_ref().unwrap())
        }
    }

    /// Returns a mutable reference to the [preconditioner](crate::pc) context set with [`KSP::set_pc()`].
    pub fn get_pc_mut<'b>(&'b mut self) -> Result<&'b mut PC<'a>>
    {
        if self.pc.is_some() {
            Ok(self.pc.as_mut().unwrap())
        } else {
            let mut pc_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::KSPGetPC(self.ksp_p, pc_p.as_mut_ptr()) };
            Petsc::check_error(self.world, ierr)?;
            let ierr = unsafe { petsc_raw::PetscObjectReference(pc_p.assume_init() as *mut petsc_raw::_p_PetscObject) };
            Petsc::check_error(self.world, ierr)?;

            self.pc = Some(PC::new(self.world, unsafe { pc_p.assume_init() }));

            Ok(self.pc.as_mut().unwrap())
        }
    }

    /// Sets the [DM](DM) that may be used by some [preconditioners](crate::pc).
    ///
    /// If this is used then the KSP will attempt to use the DM to create the matrix and use the routine
    /// set with [`DMKSPSetComputeOperators()`](#) or [`KSP::set_compute_operators()`]. Use
    /// [`KSP::set_dm_active(false)`] to instead use the matrix you've provided with [`KSP::set_operators()`].
    pub fn set_dm(&mut self, dm: DM<'a>) -> Result<()>
    {
        
        let ierr = unsafe { petsc_raw::KSPSetDM(self.ksp_p, dm.dm_p) };
        Petsc::check_error(self.world, ierr)?;

        let _ = self.dm.take();
        self.dm = Some(dm);

        Ok(())
    }

    /// Returns a reference to the [DM](DM) that may be used by some [preconditioners](crate::pc).
    pub fn get_dm<'b>(&'b mut self) -> Result<&'b DM<'a>>
    {
        if self.dm.is_some() {
            Ok(self.dm.as_ref().unwrap())
        } else {
            let mut dm_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::KSPGetDM(self.ksp_p, dm_p.as_mut_ptr()) };
            Petsc::check_error(self.world, ierr)?;
            let ierr = unsafe { petsc_raw::PetscObjectReference(dm_p.assume_init() as *mut petsc_raw::_p_PetscObject) };
            Petsc::check_error(self.world, ierr)?;

            self.dm = Some(DM::new(self.world, unsafe { dm_p.assume_init() }));

            Ok(self.dm.as_ref().unwrap())
        }
    }

    /// Returns a mutable reference to the [DM](DM) that may be used by some [preconditioners](crate::pc).
    pub fn get_dm_mut<'b>(&'b mut self) -> Result<&'b mut DM<'a>>
    {
        if self.dm.is_some() {
            Ok(self.dm.as_mut().unwrap())
        } else {
            let mut dm_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::KSPGetDM(self.ksp_p, dm_p.as_mut_ptr()) };
            Petsc::check_error(self.world, ierr)?;
            let ierr = unsafe { petsc_raw::PetscObjectReference(dm_p.assume_init() as *mut petsc_raw::_p_PetscObject) };
            Petsc::check_error(self.world, ierr)?;

            self.dm = Some(DM::new(self.world, unsafe { dm_p.assume_init() }));

            Ok(self.dm.as_mut().unwrap())
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
    pub fn set_tolerances(&mut self, rtol: Option<PetscReal>, atol: Option<PetscReal>, 
            dtol: Option<PetscReal>, max_iters: Option<PetscInt>) -> Result<()>
    {
        let ierr = unsafe { petsc_raw::KSPSetTolerances(
            self.ksp_p, rtol.unwrap_or(petsc_raw::PETSC_DEFAULT_REAL), atol.unwrap_or(petsc_raw::PETSC_DEFAULT_REAL),
            dtol.unwrap_or(petsc_raw::PETSC_DEFAULT_REAL), max_iters.unwrap_or(petsc_raw::PETSC_DEFAULT_INTEGER)) };
        Petsc::check_error(self.world, ierr)
    }

    /// Solves linear system.
    pub fn solve(&self, b: Option<&Vector>, x: &mut Vector) -> Result<()>
    {
        let ierr = unsafe { petsc_raw::KSPSolve(self.ksp_p, b.map_or(std::ptr::null_mut(), |v| v.vec_p), x.vec_p) };
        Petsc::check_error(self.world, ierr)
    }

    /// Sets the routine to compute the linear operators
    ///
    /// The user provided func() will be called automatically at the very next call to KSPSolve().
    /// It will not be called at future KSPSolve() calls unless either KSPSetComputeOperators()
    /// or KSPSetOperators() is called before that KSPSolve() is called.
    ///
    /// To reuse the same preconditioner for the next KSPSolve() and not compute a new one based
    /// on the most recently computed matrix call KSPSetReusePreconditioner()
    ///
    /// # Parameters
    ///
    /// * `user_f` - A closure used to convey the routine to compute the operators.
    ///     * `ksp` - the ksp context
    ///     * `dm` - the dm context held by the ksp
    ///     * `a_mat` *(output)* - the linear operator
    ///     * `p_mat` *(output)* - preconditioning matrix
    pub fn set_compute_operators<F>(&mut self, user_f: F) -> Result<()>
    where
        F: FnMut(&KSP<'a, 'tl>, &DM<'a>, &mut Mat<'a>, &mut Mat<'a>) -> Result<()> + 'tl
    {
        // TODO: look at how rsmpi did the trampoline stuff:
        // https://github.com/rsmpi/rsmpi/blob/master/src/collective.rs#L1684
        // They used libffi, that could be a safer way to do it.

        let closure_anchor = Box::new(user_f);

        let trampoline_data = Box::pin(KSPComputeOperatorsTrampolineData { 
            world: self.world, user_f: closure_anchor });

        // drop old trampoline_data
        let _ = self.compute_operators_trampoline_data.take();

        unsafe extern "C" fn ksp_compute_operators_trampoline(ksp_p: *mut petsc_raw::_p_KSP, mat1_p: *mut petsc_raw::_p_Mat,
            mat2_p: *mut petsc_raw::_p_Mat, ctx: *mut std::os::raw::c_void) -> petsc_raw::PetscErrorCode
        {
            // SAFETY: We construct ctx to be a Pin<Box<KSPComputeOperatorsTrampolineData>> but pass it in as a *void
            // Box<T> is equivalent to *T (or &T) for ffi. Because the KSP owns the closure we can make sure
            // everything in it (and the closure its self) lives for at least as long as this function can be
            // called.
            // We don't construct a Box<> because we dont want to drop anything
            let trampoline_data: Pin<&mut KSPComputeOperatorsTrampolineData> = std::mem::transmute(ctx);

            // We don't want to drop anything, we are just using this to turn pointers 
            // of the underlining types (i.e. *mut petsc_raw::_p_SNES) into references
            // of the rust wrapper types.
            let ksp = ManuallyDrop::new(KSP::new(trampoline_data.world, ksp_p));
            let mut dm_p = MaybeUninit::uninit();
            let _ierr = petsc_raw::KSPGetDM(ksp_p, dm_p.as_mut_ptr());
            let dm = ManuallyDrop::new(DM::new(trampoline_data.world, dm_p.assume_init()));
            let mut a_mat = ManuallyDrop::new(Mat { world: trampoline_data.world, mat_p: mat1_p });
            let mut p_mat = ManuallyDrop::new(Mat { world: trampoline_data.world, mat_p: mat2_p });
            
            (trampoline_data.get_unchecked_mut().user_f)(&ksp, &dm, &mut a_mat, &mut p_mat)
                .map_or_else(|err| err.kind as i32, |_| 0)
        }

        let ierr = unsafe { petsc_raw::KSPSetComputeOperators(
            self.ksp_p, Some(ksp_compute_operators_trampoline), 
            std::mem::transmute(trampoline_data.as_ref())) }; // this will also erase the lifetimes
        Petsc::check_error(self.world, ierr)?;
        
        self.compute_operators_trampoline_data = Some(trampoline_data);

        Ok(())
    }

    /// Sets the routine to compute the right hand side of the linear system
    ///
    /// The routine you provide will be called EACH time you call KSPSolve() to prepare the
    /// new right hand side for that solve
    ///
    /// # Parameters
    ///
    /// * `user_f` - A closure used to convey the routine to compute the the right hand side of the linear system
    ///     * `ksp` - the ksp context
    ///     * `dm` - the dm context held by the ksp
    ///     * `b` *(output)* - right hand side of linear system
    pub fn set_compute_rhs<F>(&mut self, user_f: F) -> Result<()>
    where
        F: FnMut(&KSP<'a, '_>, &DM<'a>, &mut Vector<'a>) -> Result<()> + 'tl
    {
        // TODO: look at how rsmpi did the trampoline stuff:
        // https://github.com/rsmpi/rsmpi/blob/master/src/collective.rs#L1684
        // They used libffi, that could be a safer way to do it.

        let closure_anchor = Box::new(user_f);

        let trampoline_data = Box::pin(KSPComputeRHSTrampolineData { 
            world: self.world, user_f: closure_anchor });

        // drop old trampoline_data
        let _ = self.compute_rhs_trampoline_data.take();

        unsafe extern "C" fn ksp_compute_rhs_trampoline(ksp_p: *mut petsc_raw::_p_KSP, vec_p: *mut petsc_raw::_p_Vec,
            ctx: *mut std::os::raw::c_void) -> petsc_raw::PetscErrorCode
        {

            // SAFETY: read `ksp_compute_operators_single_trampoline` safety
            let trampoline_data: Pin<&mut KSPComputeRHSTrampolineData> = std::mem::transmute(ctx);

            // We don't want to drop anything, we are just using this to turn pointers 
            // of the underlining types (i.e. *mut petsc_raw::_p_SNES) into references
            // of the rust wrapper types.
            let ksp = ManuallyDrop::new(KSP::new(trampoline_data.world, ksp_p));
            let mut dm_p = MaybeUninit::uninit();
            let _ierr = petsc_raw::KSPGetDM(ksp_p, dm_p.as_mut_ptr());
            let dm = ManuallyDrop::new(DM::new(trampoline_data.world, dm_p.assume_init()));
            let mut vec = ManuallyDrop::new(Vector { world: trampoline_data.world, vec_p });
            
            (trampoline_data.get_unchecked_mut().user_f)(&ksp, &dm, &mut vec)
                .map_or_else(|err| err.kind as i32, |_| 0)
        }

        let ierr = unsafe { petsc_raw::KSPSetComputeRHS(
            self.ksp_p, Some(ksp_compute_rhs_trampoline), 
            std::mem::transmute(trampoline_data.as_ref())) }; // this will also erase the lifetimes
        Petsc::check_error(self.world, ierr)?;
        
        self.compute_rhs_trampoline_data = Some(trampoline_data);

        Ok(())
    }

}

// macro impls
impl<'a> KSP<'a, '_> {
    wrap_simple_petsc_member_funcs! {
        KSPSetFromOptions, set_from_options, ksp_p, takes mut, #[doc = "Sets KSP options from the options database. This routine must be called before KSPSetUp() if the user is to be allowed to set the Krylov type."];
        KSPSetUp, set_up, ksp_p, takes mut, #[doc = "Sets up the internal data structures for the later use of an iterative solver. . This will be automatically called with [`KSP::solve()`]."];
        KSPGetIterationNumber, get_iteration_number, ksp_p, output PetscInt, iter_num, #[doc = "Gets the current iteration number; if the KSPSolve() is complete, returns the number of iterations used."];
    }
}

impl_petsc_object_funcs!{ KSP, ksp_p, '_ }

impl_petsc_view_func!{ KSP, ksp_p, KSPView, '_ }
