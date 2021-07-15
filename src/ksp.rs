//! The scalable linear equations solvers (KSP) component provides an easy-to-use interface to the 
//! combination of a Krylov subspace iterative method and a preconditioner (in the [KSP](crate::ksp) and [PC](crate::pc)
//! components, respectively) or a sequential direct solver. 
//! 
//! KSP users can set various Krylov subspace options at runtime via the options database 
//! (e.g., -ksp_type cg ). KSP users can also set KSP options directly in application by directly calling
//! the KSP routines listed below (e.g., [`KSP::set_type()`](#) ). KSP components can be used directly to
//! create and destroy solvers; this is not needed for users but is intended for library developers.
//!
//! PETSc C API docs: <https://petsc.org/release/docs/manualpages/KSP/index.html>

use std:: pin::Pin;
use std::mem::{MaybeUninit, ManuallyDrop};
use std::rc::Rc;
use crate::{
    Petsc,
    petsc_raw,
    Result,
    PetscAsRaw,
    PetscObject,
    PetscObjectPrivate,
    PetscReal,
    PetscInt,
    vector::{Vector, },
    mat::{Mat, },
    pc::{PC, },
    dm::{DM, },
};
use mpi::topology::UserCommunicator;
use mpi::traits::*;

/// Abstract PETSc object that manages all Krylov methods. This is the object that manages the linear
/// solves in PETSc (even those such as direct solvers that do no use Krylov accelerators).
pub struct KSP<'a, 'tl> {
    world: &'a UserCommunicator,
    pub(crate) ksp_p: *mut petsc_raw::_p_KSP, // I could use KSP which is the same thing, but i think using a pointer is more clear

    // As far as Petsc is concerned we own a reference to the PC as it is reference counted under the hood.
    // But it should be fine to keep it here as an owned reference because we can control access and the 
    // default `KSPDestroy` accounts for references.
    pc: Option<PC<'a, 'tl>>,

    dm: Option<DM<'a, 'tl>>,

    compute_operators_trampoline_data: Option<Pin<Box<KSPComputeOperatorsTrampolineData<'a, 'tl>>>>,
    compute_rhs_trampoline_data: Option<Pin<Box<KSPComputeRHSTrampolineData<'a, 'tl>>>>,
}

struct KSPComputeOperatorsTrampolineData<'a, 'tl> {
    world: &'a UserCommunicator,
    user_f: Box<dyn FnMut(&KSP<'a, 'tl>, &mut Mat<'a>, &mut Mat<'a>) -> Result<()> + 'tl>,
    set_dm: bool,
}

struct KSPComputeRHSTrampolineData<'a, 'tl> {
    world: &'a UserCommunicator,
    user_f: Box<dyn FnMut(&KSP<'a, 'tl>, &mut Vector<'a>) -> Result<()> + 'tl>,
    set_dm: bool,
}

impl<'a> Drop for KSP<'a, '_> {
    fn drop(&mut self) {
        let ierr = unsafe { petsc_raw::KSPDestroy(&mut self.ksp_p as *mut _) };
        let _ = Petsc::check_error(self.world, ierr); // TODO: should I unwrap or what idk?
    }
}

impl<'a, 'tl> KSP<'a, 'tl> {
    /// Same as `KSP { ... }` but sets all optional params to `None`
    pub(crate) fn new(world: &'a UserCommunicator, ksp_p: *mut petsc_raw::_p_KSP) -> Self {
        KSP { world, ksp_p, pc: None, dm: None,
            compute_operators_trampoline_data: None,
            compute_rhs_trampoline_data: None }
    }

    /// Same as [`Petsc::ksp_create()`].
    pub fn create(world: &'a UserCommunicator) -> Result<Self> {
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
    pub fn set_operators(&mut self, a_mat: impl Into<Option<Rc<Mat<'a>>>>, p_mat: impl Into<Option<Rc<Mat<'a>>>>) -> Result<()>
    {
        let a_mat = a_mat.into();
        let p_mat = p_mat.into();
        // TODO: should we call `KSPSetOperators`? or should we just call `PC::set_operators`
        // The source for KSPSetOperators basically just calls PCSetOperators but does something with 
        // `ksp->setupstage` so idk.
        self.get_pc_mut()?.set_operators(a_mat, p_mat)
    }

    /// Sets the [preconditioner](crate::pc)([`PC`]) to be used to calculate the application of
    /// the preconditioner on a vector.
    ///
    /// if you change the PC by calling set again, then the original will be dropped.
    pub fn set_pc(&mut self, pc: PC<'a, 'tl>) -> Result<()>
    {
        
        let ierr = unsafe { petsc_raw::KSPSetPC(self.ksp_p, pc.pc_p) };
        Petsc::check_error(self.world, ierr)?;

        let _ = self.pc.take();
        self.pc = Some(pc);

        Ok(())
    }
    
    /// Returns a [`Option`] of a reference to the [`PC`] context set.
    ///
    /// If you want PETSc to set the [`PC`] you must call [`KSP::get_pc()`]
    /// or [`KSP::get_pc_mut()`].
    ///
    /// Note, this does not return a [`Result`](crate::Result) because it can never
    /// fail, instead it will return `None`.
    pub fn try_get_pc<'b>(&'b self) -> Option<&'b PC<'a, 'tl>> {
        self.pc.as_ref()
    }

    /// Returns a reference to the [`PC`] context set with [`KSP::set_pc()`].
    pub fn get_pc<'b>(&'b mut self) -> Result<&'b PC<'a, 'tl>>
    {
        // TODO: This shouldn't have to take a mut self (maybe we use a RefCell internally)

        // Under the hood, if the pc is already set, i.e. with `set_pc`, then `KSPGetPC` just returns a pointer
        // to that, so we can bypass calling KSPGetPC. However, there shouldn't be any problem with just calling
        // `KSPGetPC` again as we incremented the reference of the PC we "own" so dropping it wont do anything.
        if self.pc.is_some() {
            Ok(self.pc.as_ref().unwrap())
        } else {
            let mut pc_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::KSPGetPC(self.ksp_p, pc_p.as_mut_ptr()) };
            Petsc::check_error(self.world, ierr)?;

            self.pc = Some(PC::new(self.world, unsafe { pc_p.assume_init() }));
            unsafe { self.pc.as_mut().unwrap().reference()?; }

            Ok(self.pc.as_ref().unwrap())
        }
    }

    /// Returns a mutable reference to the [`PC`] context set with [`KSP::set_pc()`].
    pub fn get_pc_mut<'b>(&'b mut self) -> Result<&'b mut PC<'a, 'tl>>
    {
        if self.pc.is_some() {
            Ok(self.pc.as_mut().unwrap())
        } else {
            let mut pc_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::KSPGetPC(self.ksp_p, pc_p.as_mut_ptr()) };
            Petsc::check_error(self.world, ierr)?;

            self.pc = Some(PC::new(self.world, unsafe { pc_p.assume_init() }));
            unsafe { self.pc.as_mut().unwrap().reference()?; }

            Ok(self.pc.as_mut().unwrap())
        }
    }

    /// Sets the [DM](DM) that may be used by some [preconditioners](crate::pc).
    ///
    /// If this is used then the KSP will attempt to use the DM to create the matrix and use the routine
    /// set with [`DMKSPSetComputeOperators()`](#) or [`KSP::set_compute_operators()`]. Use
    /// [`KSP::set_dm_active(false)`] to instead use the matrix you've provided with [`KSP::set_operators()`].
    pub fn set_dm(&mut self, dm: DM<'a, 'tl>) -> Result<()>
    {
        
        let ierr = unsafe { petsc_raw::KSPSetDM(self.ksp_p, dm.dm_p) };
        Petsc::check_error(self.world, ierr)?;

        let _ = self.dm.take();
        self.dm = Some(dm);

        Ok(())
    }

    /// Returns an [`Option`] to a reference to the [DM](DM).
    ///
    /// If you want PETSc to set the [`DM`] you must call [`KSP::get_dm()`]
    /// or [`KSP::get_dm_mut()`], otherwise you must call [`KSP::set_dm()`]
    /// for this to return a `Some`.
    ///
    /// Note, this does not return a [`Result`](crate::Result) because it can never
    /// fail, instead it will return `None`.
    pub fn try_get_dm<'b>(&'b self) -> Option<&'b DM<'a, 'tl>> {
        self.dm.as_ref()
    }

    /// Returns a reference to the [DM](DM) that may be used by some [preconditioners](crate::pc).
    pub fn get_dm<'b>(&'b mut self) -> Result<&'b DM<'a, 'tl>>
    {
        if self.dm.is_some() {
            Ok(self.dm.as_ref().unwrap())
        } else {
            let mut dm_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::KSPGetDM(self.ksp_p, dm_p.as_mut_ptr()) };
            Petsc::check_error(self.world, ierr)?;

            self.dm = Some(DM::new(self.world, unsafe { dm_p.assume_init() }));
            unsafe { self.dm.as_mut().unwrap().reference()?; }

            Ok(self.dm.as_ref().unwrap())
        }
    }

    /// Returns a mutable reference to the [DM](DM) that may be used by some [preconditioners](crate::pc).
    pub fn get_dm_mut<'b>(&'b mut self) -> Result<&'b mut DM<'a, 'tl>>
    {
        if self.dm.is_some() {
            Ok(self.dm.as_mut().unwrap())
        } else {
            let mut dm_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::KSPGetDM(self.ksp_p, dm_p.as_mut_ptr()) };
            Petsc::check_error(self.world, ierr)?;

            self.dm = Some(DM::new(self.world, unsafe { dm_p.assume_init() }));
            unsafe { self.dm.as_mut().unwrap().reference()?; }

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
    pub fn set_tolerances(&mut self, rtol: impl Into<Option<PetscReal>>, atol: impl Into<Option<PetscReal>>, 
            dtol: impl Into<Option<PetscReal>>, max_iters: impl Into<Option<PetscInt>>) -> Result<()>
    {
        let ierr = unsafe { petsc_raw::KSPSetTolerances(
            self.ksp_p, rtol.into().unwrap_or(petsc_raw::PETSC_DEFAULT_REAL), atol.into().unwrap_or(petsc_raw::PETSC_DEFAULT_REAL),
            dtol.into().unwrap_or(petsc_raw::PETSC_DEFAULT_REAL), max_iters.into().unwrap_or(petsc_raw::PETSC_DEFAULT_INTEGER)) };
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
    ///     * `a_mat` *(output)* - the linear operator
    ///     * `p_mat` *(output)* - preconditioning matrix
    ///
    /// # Note
    ///
    /// You can access the [`DM`] owned by the `ksp` in the `user_f` by using
    /// [`let dm = ksp.try_get_dm().unwrap();`](KSP::try_get_dm()). Note, this will only work
    /// if you set the dm with [`KSP::set_dm()()`] BEFORE you call the
    /// [`set_compute_operators()`](KSP::set_compute_operators) method.
    pub fn set_compute_operators<F>(&mut self, user_f: F) -> Result<()>
    where
        F: FnMut(&KSP<'a, 'tl>, &mut Mat<'a>, &mut Mat<'a>) -> Result<()> + 'tl
    {
        // TODO: look at how rsmpi did the trampoline stuff:
        // https://github.com/rsmpi/rsmpi/blob/master/src/collective.rs#L1684
        // They used libffi, that could be a safer way to do it.

        let closure_anchor = Box::new(user_f);

        // Note, if the caller has previously called `KSP::set_dm()` then we want to make
        // sure to give them access in the closure.
        // TODO: it would be nice to determine this in the trampoline function as the caller could
        // set the dm after they set this closure (for now i think this is fine)
        let trampoline_data = Box::pin(KSPComputeOperatorsTrampolineData { 
            world: self.world, user_f: closure_anchor, set_dm: self.dm.is_some() });

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
            let mut ksp = ManuallyDrop::new(KSP::new(trampoline_data.world, ksp_p));
            if trampoline_data.set_dm {
                let mut dm_p = MaybeUninit::uninit();
                let ierr = petsc_raw::KSPGetDM(ksp_p, dm_p.as_mut_ptr());
                if ierr != 0 { let _ = Petsc::check_error(trampoline_data.world, ierr); return ierr; }
                let dm = DM::new(trampoline_data.world, dm_p.assume_init());
                ksp.dm = Some(dm); // Note, because ksp is not dropped, ksp.dm wont be either
            }
            let mut a_mat = ManuallyDrop::new(Mat { world: trampoline_data.world, mat_p: mat1_p });
            let mut p_mat = ManuallyDrop::new(Mat { world: trampoline_data.world, mat_p: mat2_p });
            
            (trampoline_data.get_unchecked_mut().user_f)(&ksp, &mut a_mat, &mut p_mat)
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
    ///
    /// # Note
    ///
    /// You can access the [`DM`] owned by the `ksp` in the `user_f` by using
    /// [`let dm = ksp.try_get_dm().unwrap();`](KSP::try_get_dm()). Note, this will only work
    /// if you set the dm with [`KSP::set_dm()()`] BEFORE you call the
    /// [`set_compute_rhs()`](KSP::set_compute_rhs) method.
    pub fn set_compute_rhs<F>(&mut self, user_f: F) -> Result<()>
    where
        F: FnMut(&KSP<'a, '_>, &mut Vector<'a>) -> Result<()> + 'tl
    {
        // TODO: look at how rsmpi did the trampoline stuff:
        // https://github.com/rsmpi/rsmpi/blob/master/src/collective.rs#L1684
        // They used libffi, that could be a safer way to do it.

        let closure_anchor = Box::new(user_f);

        // Note, if the caller has previously called `KSP::set_dm()` then we want to make
        // sure to give them access in the closure.
        // TODO: it would be nice to determine this in the trampoline function as the caller could
        // set the dm after they set this closure (for now i think this is fine)
        let trampoline_data = Box::pin(KSPComputeRHSTrampolineData { 
            world: self.world, user_f: closure_anchor, set_dm: self.dm.is_some() });

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
            let mut ksp = ManuallyDrop::new(KSP::new(trampoline_data.world, ksp_p));
            if trampoline_data.set_dm {
                let mut dm_p = MaybeUninit::uninit();
                let ierr = petsc_raw::KSPGetDM(ksp_p, dm_p.as_mut_ptr());
                if ierr != 0 { let _ = Petsc::check_error(trampoline_data.world, ierr); return ierr; }
                let dm = DM::new(trampoline_data.world, dm_p.assume_init());
                ksp.dm = Some(dm); // Note, because ksp is not dropped, ksp.dm wont be either
            }
            let mut vec = ManuallyDrop::new(Vector { world: trampoline_data.world, vec_p });
            
            (trampoline_data.get_unchecked_mut().user_f)(&ksp, &mut vec)
                .map_or_else(|err| err.kind as i32, |_| 0)
        }

        let ierr = unsafe { petsc_raw::KSPSetComputeRHS(
            self.ksp_p, Some(ksp_compute_rhs_trampoline), 
            std::mem::transmute(trampoline_data.as_ref())) }; // this will also erase the lifetimes
        Petsc::check_error(self.world, ierr)?;
        
        self.compute_rhs_trampoline_data = Some(trampoline_data);

        Ok(())
    }

    /// Gets the right-hand-side vector for the linear system to be solved.
    pub fn get_rhs(&self) -> Result<Rc<Vector<'a>>> {
        let mut vec_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::KSPGetRhs(self.ksp_p, vec_p.as_mut_ptr()) };
        Petsc::check_error(self.world, ierr)?;

        let mut vec = Vector { world: self.world, vec_p: unsafe { vec_p.assume_init() } };
        unsafe { vec.reference()?; }
        let rhs = Rc::new(vec);

        Ok(rhs)
    }

    /// Gets the matrix associated with the linear system and possibly a different
    /// one associated with the preconditioner.
    ///
    /// See also [`PC::get_operators()`].
    ///
    /// If the operators have NOT been set with [`KSP`](KSP::set_operators())/[`PC::set_operators()`]
    /// then the operators are created in the PC and returned to the user. In this case, two DIFFERENT
    /// operators will be returned.
    pub fn get_operators(&mut self) -> Result<(Rc<Mat<'a>>, Rc<Mat<'a>>)> {
        // TODO: this shouldn't have to take `&mut self`
        self.get_pc()?.get_operators()
    }
}

// macro impls
impl<'a> KSP<'a, '_> {
    wrap_simple_petsc_member_funcs! {
        KSPSetFromOptions, pub set_from_options, takes mut, #[doc = "Sets KSP options from the options database. This routine must be called before KSPSetUp() if the user is to be allowed to set the Krylov type."];
        KSPSetUp, pub set_up, takes mut, #[doc = "Sets up the internal data structures for the later use of an iterative solver. . This will be automatically called with [`KSP::solve()`]."];
        KSPGetIterationNumber, pub get_iteration_number, output PetscInt, iter_num, #[doc = "Gets the current iteration number; if the KSPSolve() is complete, returns the number of iterations used."];
    }
}

impl_petsc_object_traits! { KSP, ksp_p, petsc_raw::_p_KSP, '_ }

impl_petsc_view_func!{ KSP, KSPView, '_ }
