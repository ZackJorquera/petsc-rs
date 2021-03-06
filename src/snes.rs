//! The Scalable Nonlinear Equations Solvers (SNES) component provides an easy-to-use interface to
//! Newton-type, quasi-Newton, full approximation scheme (FAS) multigrid, and other methods for solving
//! systems of nonlinear equations. 
//!
//! SNES users can set various algorithmic options at runtime via the
//! options database (e.g., specifying a trust region method via -snes_type newtontr ). SNES internally
//! employs [KSP](crate::ksp) for the solution of its linear systems. SNES users can also set [`KSP`](KSP) options
//! directly in application codes by first extracting the [`KSP`](KSP) context from the [`SNES`](crate::snes::SNES) context via
//! [`SNES::get_ksp_or_create()`] and then directly calling various [`KSP`](KSP) (and [`PC`](crate::pc::PC)) routines
//! (e.g., [`PC::set_type()`](#)).
//!
//! PETSc C API docs: <https://petsc.org/release/docs/manualpages/SNES/index.html>

use std::ops::DerefMut;
use std::pin::Pin;
use std::mem::{MaybeUninit, ManuallyDrop};
use std::ffi::CStr;
use std::slice;
use crate::{
    petsc_raw,
    Result,
    PetscAsRaw,
    PetscObject,
    PetscObjectPrivate,
    PetscReal,
    PetscInt,
    vector::{Vector, },
    mat::{Mat, },
    ksp::{KSP, },
    dm::{DM, },
};
use mpi::topology::UserCommunicator;
use mpi::traits::*;


/// [`SNES`] Type
pub use crate::petsc_raw::SNESTypeEnum as SNESType;

/// Abstract PETSc object that manages all nonlinear solves
pub struct SNES<'a, 'tl, 'bl> {
    world: &'a UserCommunicator,
    pub(crate) snes_p: *mut petsc_raw::_p_SNES,

    ksp: Option<KSP<'a, 'tl, 'bl>>,
    linesearch: Option<LineSearch<'a>>,
    pub(crate) dm: Option<DM<'a, 'tl>>,

    residual_vec: Option<&'bl mut Vector<'a>>,
    residual_vec_owned: Option<Vector<'a>>,
    jacobian_a_mat: Option<&'bl mut Mat<'a, 'tl>>,
    jacobian_p_mat: Option<&'bl mut Mat<'a, 'tl>>,

    monitor_tramoline_data: Option<Pin<Box<SNESMonitorTrampolineData<'a,'tl>>>>,
    
    linecheck_post_check_trampoline_data: Option<Pin<Box<SNESLineSearchPostCheckTrampolineData<'a, 'tl>>>>,
    linecheck_pre_check_trampoline_data: Option<Pin<Box<SNESLineSearchPreCheckTrampolineData<'a, 'tl>>>>,
}

// TODO: the linesearch api needs work (i don't really like it). Regardless, the C api of it is not safe
// for rust (and more than most things). This is because a linesearch is owned by a SNES. But then when
// you use the line search you use `SNESLineSearchGetSNES` meaning that we have a self referential struct.
// I see two ways to do this in rust (to have a safe(ish) maybe)) api:
// 1. We dont give the `LineSearch` access to a SNES on the rust side and where make all methods that require
// as SNES be called through the SNES (Note: under the hood `SNESLineSearchGetSNES` might still be used as
// this layout is just to enforced ownership and stuff). This is the method im using
// 2. we make linesearch be above the SNES. This has a lot of problem and i dont think it make sense for PETSc
/// Abstract PETSc object that manages line-search operations 
pub struct LineSearch <'a> {
    world: &'a UserCommunicator,
    ls_p: *mut petsc_sys::_p_LineSearch,
}

/// A PETSc error type with a [`DomainErr`](DomainOrPetscError::DomainErr) variant.
///
/// Implements [`From<PetscError>`](From), so it works nicely with the try operator, `?`,
/// on [`Err(PetscError)`](crate::PetscError).
///
/// Used with function that compute operators like [`SNES::set_function()`] or [`SNES::set_jacobian()`]
/// That can have the input out of the domain.
pub enum DomainOrPetscError {
    /// Used to indicate that there was a domain error.
    ///
    /// This will not create a `PetscError` internally unless you specify that there should be an 
    /// error if not converged (i.e. with [`SNES::set_error_if_not_converged()`]).
    DomainErr,
    /// Normal PetscError.
    ///
    /// You should not need to create this variant as [`DomainOrPetscError`]
    /// implements [`From<PetscError>`](From), so it works nicely with the try operator, `?`,
    /// on [`Err(PetscError)`](crate::PetscError).
    PetscErr(crate::PetscError)
}

impl From<crate::PetscError> for DomainOrPetscError {
    fn from(pe: crate::PetscError) -> DomainOrPetscError {
        DomainOrPetscError::PetscErr(pe)
    }
}

struct SNESMonitorTrampolineData<'a, 'tl> {
    world: &'a UserCommunicator,
    user_f: Box<dyn FnMut(&SNES<'a, 'tl, '_>, PetscInt, PetscReal) -> Result<()> + 'tl>,
}

struct SNESLineSearchPostCheckTrampolineData<'a, 'tl> {
    world: &'a UserCommunicator,
    user_f: Box<dyn FnMut(&LineSearch<'a>, &SNES<'a, 'tl, '_>, &Vector<'a>, &mut Vector<'a>, &mut Vector<'a>, &mut bool, &mut bool) -> Result<()> + 'tl>,
}

struct SNESLineSearchPreCheckTrampolineData<'a, 'tl> {
    world: &'a UserCommunicator,
    user_f: Box<dyn FnMut(&LineSearch<'a>, &SNES<'a, 'tl, '_>, &Vector<'a>, &mut Vector<'a>, &mut bool) -> Result<()> + 'tl>,
}

/// Reason a [`SNES`] method was said to have converged or diverged.
///
/// Also read: <https://petsc.org/release/docs/manualpages/SNES/SNESConvergedReason.html>
pub use petsc_raw::SNESConvergedReason;

impl<'a, 'tl, 'bl> SNES<'a, 'tl, 'bl> {
    /// Same as `SNES { ... }` but sets all optional params to `None`
    pub(crate) fn new(world: &'a UserCommunicator, snes_p: *mut petsc_raw::_p_SNES) -> Self {
        SNES {  world, snes_p, ksp: None, monitor_tramoline_data: None,
                linesearch: None, dm: None,
                linecheck_post_check_trampoline_data: None,
                linecheck_pre_check_trampoline_data: None,
                residual_vec: None, jacobian_a_mat: None,
                jacobian_p_mat: None, residual_vec_owned: None, }
    }

    /// Creates a nonlinear solver context.
    pub fn create(world: &'a UserCommunicator) -> Result<Self> {
        let mut snes_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::SNESCreate(world.as_raw(), snes_p.as_mut_ptr()) };
        unsafe { chkerrq!(world, ierr) }?;

        Ok(SNES::new(world, unsafe { snes_p.assume_init() }))
    }

    /// Sets a [`KSP`](KSP) context for the SNES object to use.
    ///
    /// if you change the ksp by calling set again, then the original will be dropped.
    pub fn set_ksp(&mut self, ksp: KSP<'a, 'tl, 'bl>) -> Result<()>
    {
        
        let ierr = unsafe { petsc_raw::SNESSetKSP(self.snes_p, ksp.ksp_p) };
        unsafe { chkerrq!(self.world, ierr) }?;

        let _ = self.ksp.take();
        self.ksp = Some(ksp);

        Ok(())
    }

    /// Returns an [`Option`] to a reference to the [`KSP`](KSP) context.
    ///
    /// If you want PETSc to set the [`KSP`] you must call [`SNES::set_ksp()`]
    /// or [`SNES::get_ksp_or_create()`].
    ///
    /// Note, this does not return a [`Result`](crate::Result) because it can never
    /// fail, instead it will return `None`.
    pub fn try_get_ksp(&self) -> Option<&KSP<'a, 'tl, 'bl>> {
        self.ksp.as_ref()
    }

    /// Returns a mutable reference to the [`KSP`](KSP) context, or create a default [`KSP`](KSP)
    /// if one has not been set.
    pub fn get_ksp_or_create(&mut self) -> Result<&mut KSP<'a, 'tl, 'bl>> {
        if let Some(ref mut ksp) = self.ksp
        {
            Ok(ksp)
        }
        else
        {
            let mut ksp_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::SNESGetKSP(self.snes_p, ksp_p.as_mut_ptr()) };
            unsafe { chkerrq!(self.world, ierr) }?;

            // It is now ok to drop this because we incremented the C reference counter
            self.ksp = Some(KSP::new(self.world, unsafe { ksp_p.assume_init() }));
            unsafe { self.ksp.as_mut().unwrap().reference()?; }

            Ok(self.ksp.as_mut().unwrap())
        }
    }

    /// Sets the function evaluation routine and function vector for use by the [`SNES`] routines in solving
    /// systems of nonlinear equations.
    ///
    /// # Parameters
    ///
    /// * `input_vec` - vector to store function value (if None, the SNES DM will create it).
    /// * `user_f` - A closure used to convey the nonlinear function to be solved by SNES
    ///     * `snes` - the snes context
    ///     * `x` - state at which to evaluate residual
    ///     * `f` *(output)* - vector to put residual (function value)
    ///
    /// # Note
    ///
    /// The Newton-like methods typically solve linear systems of the form
    /// ```text
    /// f'(x) x = -f(x),
    /// ```
    /// where `f'(x)` denotes the Jacobian matrix and `f(x)` is the function.
    ///
    /// You can access the [`DM`] owned by the `snes` in the `user_f` by using
    /// [`let dm = snes.try_get_dm().unwrap();`](SNES::try_get_dm()).
    ///
    /// # Example
    ///
    /// See example: `snes-ex2` for full code (at 
    /// [`examples/snes/src/ex2.rs`](https://gitlab.com/petsc/petsc-rs/-/blob/main/examples/snes/src/ex2.rs)).
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # use mpi::traits::*;
    /// # fn main() -> petsc_rs::Result<()> {
    /// # let petsc = Petsc::init_no_args()?;
    /// let n = 10;
    /// let mut g = petsc.vec_create()?;
    /// g.set_sizes(None, n)?;
    /// g.set_from_options()?;
    /// let mut r = g.duplicate()?;
    ///
    /// let mut snes = petsc.snes_create()?;
    ///
    /// snes.set_function(&mut r, |_snes, x: &Vector, f: &mut Vector| {
    ///     let x_view = x.view()?;
    ///     let mut f_view = f.view_mut()?;
    ///     let g_view = g.view()?;
    ///
    ///     let d = (PetscScalar::from(n as PetscReal) - 1.0).powi(2);
    ///
    ///     // Nonlinear transformation
    ///     f_view[0] = x_view[0];
    ///     for i in 1..(n as usize - 1) {
    ///         f_view[i] = d*(x_view[i-1] - 2.0*x_view[i] + x_view[i+1]) + x_view[i]*x_view[i] - g_view[i];
    ///     }
    ///     f_view[n as usize - 1] = x_view[n as usize - 1] - 1.0;
    ///
    ///     Ok(())
    /// })?;
    ///
    /// // We would then set the jacobian and call solve on the snes context
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_function<F>(&mut self, input_vec: impl Into<Option<&'bl mut Vector<'a>>>, user_f: F) -> Result<()>
    where
        F: FnMut(&SNES<'a, '_, '_>, &Vector<'a>, &mut Vector<'a>) -> std::result::Result<(), DomainOrPetscError> + 'tl
    {
        self.residual_vec = input_vec.into();

        let input_vec_p = self.residual_vec.as_deref().map_or(std::ptr::null_mut(), |v| v.vec_p);

        let ierr = unsafe { petsc_raw::SNESSetFunction(
            self.snes_p, input_vec_p, None, std::ptr::null_mut()) };
        unsafe { chkerrq!(self.world, ierr) }?;

        self.get_dm_or_create()?.snes_set_function(user_f)
    }

    /// Sets the function to compute Jacobian as well as the location to store the matrix.
    ///
    /// Allows you to set a function to define what the Jacobian matrix is.
    ///
    /// # Parameters
    ///
    /// * `ap_mat` - the matrix to be used in constructing the (approximate) Jacobian as well as
    /// the preconditioner. If you wish these to be different matrices use the function
    /// [`SNES::set_jacobian()`] instead.
    /// * `user_f` - A closure used to convey the Jacobian evaluation routine.
    ///     * `snes` - the snes context
    ///     * `x` - input vector, the Jacobian is to be computed at this value
    ///     * `ap_mat` *(output)* - the matrix to be used in constructing the (approximate) Jacobian as well as
    ///     the preconditioner.
    ///
    /// # Note
    ///
    /// You are expected to call [`Mat::assembly_begin()`] and [`Mat::assembly_end()`] at the end of
    /// `user_f`. Or you can something like [`Mat::assemble_with()`].
    ///
    /// You can access the [`DM`] owned by the `snes` in the `user_f` by using
    /// [`let dm = snes.try_get_dm().unwrap();`](SNES::try_get_dm()).
    ///
    /// # Example
    ///
    /// See example: `snes-ex2` for full code (at 
    /// [`examples/snes/src/ex2.rs`](https://gitlab.com/petsc/petsc-rs/-/blob/main/examples/snes/src/ex2.rs)).
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # use mpi::traits::*;
    /// # fn main() -> petsc_rs::Result<()> {
    /// # let petsc = Petsc::init_no_args()?;
    /// let n = 10;
    ///
    /// # #[allow(non_snake_case)]
    /// let mut J = petsc.mat_create()?;
    /// J.set_sizes(None, None, Some(n), Some(n))?;
    /// J.set_from_options()?;
    /// J.seq_aij_set_preallocation(3, None)?;
    ///
    /// let mut snes = petsc.snes_create()?;
    ///
    /// snes.set_jacobian_single_mat(&mut J,|_snes, x: &Vector, ap_mat: &mut Mat| {
    ///     let x_view = x.view()?;
    ///
    ///     let d = (PetscScalar::from(n as PetscReal) - 1.0).powi(2);
    ///
    ///     ap_mat.assemble_with((0..n).map(|i| if i == 0 || i == n-1{ vec![(i,i,PetscScalar::from(1.0))] }
    ///                                         else { vec![(i,i-1,d), (i,i,-2.0*d+2.0*x_view[i as usize]), (i,i+1,d)] })
    ///             .flatten(), 
    ///         InsertMode::INSERT_VALUES, MatAssemblyType::MAT_FINAL_ASSEMBLY)?;
    ///
    ///     Ok(())
    /// })?;
    ///
    /// // We would then set the function and call solve on the snes context
    /// # Ok(())
    /// # }
    /// ```
    // pub fn set_jacobian_single_mat<F>(&mut self, ap_mat: Mat<'a, 'tl>, user_f: F) -> Result<()>
    pub fn set_jacobian_single_mat<F>(&mut self, ap_mat: &'bl mut Mat<'a, 'tl>, user_f: F) -> Result<()>
    where
        F: FnMut(&SNES<'a, '_, '_>, &Vector<'a>, &mut Mat<'a, '_>) -> std::result::Result<(), DomainOrPetscError> + 'tl,
    {
        self.jacobian_a_mat = Some(ap_mat);
        let ap_mat_p = self.jacobian_a_mat.as_deref().unwrap().mat_p;

        let ierr = unsafe { petsc_raw::SNESSetJacobian(
            self.snes_p, ap_mat_p, ap_mat_p, None, 
            std::ptr::null_mut()) };
        unsafe { chkerrq!(self.world, ierr) }?;

        self.get_dm_or_create()?.snes_set_jacobian_single_mat(user_f)
    }

    /// Sets the function to compute Jacobian as well as the location to store the matrix.
    ///
    /// Allows you to set a function to define what the Jacobian matrix is and what the preconditioner
    /// matrix is separately.
    ///
    /// # Parameters
    ///
    /// * `a_mat` - the matrix that defines the (approximate) Jacobian
    /// * `p_mat` - the matrix to be used in constructing the preconditioner. If you wish to use the
    /// same matrix as `a_mat` then you need to use the [`SNES::set_jacobian_single_mat()`] instead.
    /// * `user_f` - A closure used to convey the Jacobian evaluation routine.
    ///     * `snes` - the snes context
    ///     * `x` - input vector, the Jacobian is to be computed at this value
    ///     * `a_mat` *(output)* - the matrix that defines the (approximate) Jacobian.
    ///     * `p_mat` *(output)* - the matrix to be used in constructing the preconditioner.
    ///
    /// # Note
    ///
    /// You are expected to call [`Mat::assembly_begin()`] and [`Mat::assembly_end()`] at the end of
    /// `user_f` on both `a_mat` and `p_mat`. Or you can [`Mat::assemble_with()`].
    ///
    /// You can access the [`DM`] owned by the `snes` in the `user_f` by using
    /// [`let dm = snes.try_get_dm().unwrap();`](SNES::try_get_dm()).
    ///
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # use mpi::traits::*;
    /// # fn main() -> petsc_rs::Result<()> {
    /// # let petsc = Petsc::init_no_args()?;
    /// let n = 10;
    ///
    /// # #[allow(non_snake_case)]
    /// let mut J = petsc.mat_create()?;
    /// J.set_sizes(None, None, Some(n), Some(n))?;
    /// J.set_from_options()?;
    /// J.seq_aij_set_preallocation(3, None)?;
    /// # #[allow(non_snake_case)]
    /// let mut P = petsc.mat_create()?;
    /// P.set_sizes(None, None, Some(n), Some(n))?;
    /// P.set_from_options()?;
    /// P.seq_aij_set_preallocation(3, None)?;
    ///
    /// let mut snes = petsc.snes_create()?;
    ///
    /// snes.set_jacobian(&mut J, &mut P,|_snes, x: &Vector, a_mat: &mut Mat, p_mat: &mut Mat| {
    ///     let x_view = x.view()?;
    ///
    ///     let d = (PetscScalar::from(n as PetscReal) - 1.0).powi(2);
    ///
    ///     a_mat.assemble_with((0..n).map(|i| if i == 0 || i == n-1{ vec![(i,i,PetscScalar::from(1.0))] }
    ///                                         else { vec![(i,i-1,d), (i,i,-2.0*d+2.0*x_view[i as usize]), (i,i+1,d)] })
    ///             .flatten(), 
    ///         InsertMode::INSERT_VALUES, MatAssemblyType::MAT_FINAL_ASSEMBLY)?;
    ///
    ///     // Just set them to be the same matrix for the sake of this example
    ///     p_mat.assemble_with((0..n).map(|i| if i == 0 || i == n-1{ vec![(i,i,PetscScalar::from(1.0))] }
    ///                                         else { vec![(i,i-1,d), (i,i,-2.0*d+2.0*x_view[i as usize]), (i,i+1,d)] })
    ///             .flatten(), 
    ///         InsertMode::INSERT_VALUES, MatAssemblyType::MAT_FINAL_ASSEMBLY)?;
    ///
    ///     Ok(())
    /// })?;
    ///
    /// // We would then set the function and call solve on the snes context
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_jacobian<F>(&mut self, a_mat: &'bl mut Mat<'a, 'tl>, p_mat: &'bl mut Mat<'a, 'tl>, user_f: F) -> Result<()>
    where
        F: FnMut(&SNES<'a, '_, '_>, &Vector<'a>, &mut Mat<'a, '_>, &mut Mat<'a, '_>) -> std::result::Result<(), DomainOrPetscError> + 'tl
    {
        self.jacobian_a_mat = Some(a_mat);
        self.jacobian_p_mat = Some(p_mat);
        let a_mat_p = self.jacobian_a_mat.as_deref().unwrap().mat_p;
        let p_mat_p = self.jacobian_p_mat.as_deref().unwrap().mat_p;

        let ierr = unsafe { petsc_raw::SNESSetJacobian(
            self.snes_p, a_mat_p, p_mat_p, None,
            std::ptr::null_mut()) };
        unsafe { chkerrq!(self.world, ierr) }?;
        
        self.get_dm_or_create()?.snes_set_jacobian(user_f)
    }

    /// Sets an ADDITIONAL function that is to be used at every iteration of the nonlinear
    /// solver to display the iteration's progress.
    ///
    /// Several different monitoring routines may be set by calling [`SNES::monitor_set()`]
    /// multiple times; all will be called in the order in which they were set. 
    ///
    /// # Parameters
    ///
    /// * `user_f` - A closure used to convey the monitor function.
    ///     * `snes` - the snes context
    ///     * `it` - iteration number
    ///     * `norm - 2-norm function value (may be estimated)
    pub fn monitor_set<F>(&mut self, user_f: F) -> Result<()>
    where
        F: FnMut(&SNES<'a, 'tl, '_>, PetscInt, PetscReal) -> Result<()> + 'tl
    {
        let closure_anchor = Box::new(user_f);

        let trampoline_data = Box::pin(SNESMonitorTrampolineData { 
            world: self.world, user_f: closure_anchor });
        let _ = self.monitor_tramoline_data.take();

        unsafe extern "C" fn snes_monitor_trampoline(snes_p: *mut petsc_raw::_p_SNES,
            it: PetscInt, norm: PetscReal, ctx: *mut ::std::os::raw::c_void) -> petsc_raw::PetscErrorCode
        {
            // SAFETY: read `snes_function_trampoline` safety
            let trampoline_data: Pin<&mut SNESMonitorTrampolineData> = std::mem::transmute(ctx);

            // We don't want to drop anything, we are just using this to turn pointers 
            // of the underlining types (i.e. *mut petsc_raw::_p_SNES) into references.
            // SAFETY: even though snes is mut and thus we can set optional parameters, we don't
            // as we dont expose the mut to the user closure, we only use it with `set_jacobian_domain_error`
            let snes = ManuallyDrop::new(SNES::new(trampoline_data.world, snes_p));
            
            (trampoline_data.get_mut().user_f)(&snes, it, norm)
                .map_or_else(|err| err.kind as i32, |_| 0)
        }

        let ierr = unsafe { petsc_raw::SNESMonitorSet(
            self.snes_p, Some(snes_monitor_trampoline), 
            std::mem::transmute(trampoline_data.as_ref()), 
            None) }; // We dont need to tell C the drop function because rust will take care of it for us.
        unsafe { chkerrq!(self.world, ierr) }?;
        
        self.monitor_tramoline_data = Some(trampoline_data);

        Ok(())
    }

    /// Returns an immutable reference to the line search context set with `SNESSetLineSearch()`.
    pub fn try_get_linesearch(&self) -> Option<&LineSearch<'a>> {
        self.linesearch.as_ref()
    }

    /// Returns a mutable reference to the line search context set with `SNESSetLineSearch()`
    /// or creates a default line search instance associated with the SNES and returns it. 
    pub fn get_linesearch_or_create(&mut self) -> Result<&mut LineSearch<'a>> {
        if self.linesearch.is_some() {
            Ok(self.linesearch.as_mut().unwrap())
        } else {
            let mut ls_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::SNESGetLineSearch(self.snes_p, ls_p.as_mut_ptr()) };
            unsafe { chkerrq!(self.world, ierr) }?;

            self.linesearch = Some(LineSearch { world: self.world, ls_p: unsafe { ls_p.assume_init() } });
            unsafe { self.linesearch.as_mut().unwrap().reference()?; }

            Ok(self.linesearch.as_mut().unwrap())
        }
    }

    // TODO: we can do better that having the user say if they edited the vec (how does Cow work? can we use that here?)
    // Also note that if we are running on multiple processes, the changed flag must be the same for all processes.
    // Look into `ndarray::CowArray`, would that help?
    /// Sets a user function that is called after the line search has been applied to determine the step
    /// direction and length. Allows the user to change or override the decision of the line search routine.
    ///
    /// # Parameters 
    ///
    /// * `user_f` - function evaluation routine
    ///     * `ls` - The linesearch context 
    ///     * `snes` - The snes context 
    ///     * `x` - The last step
    ///     * `y` - The mutable step direction.
    ///     * `w` - The mutable updated solution, normally `w = x + lambda*y` for some lambda.
    ///     * `changed_y` *(output)* - Indicator if the direction `y` has been changed.
    ///     * `changed_w` *(output)* - Indicator if the new candidate solution `w` has been changed.
    ///
    /// # Note
    ///
    /// Note, when the `post_check` function is called, `petsc-rs` will automatically apply
    /// a logical OR via an MPI all reduce on the `change_*` values. That is to say, if you
    /// set one of the change values to `true` on any process, it will be treated as if all
    /// processes returned true. This is different from the C API which requires all the
    /// values to be logically collective (the same for each process).
    ///
    /// You can access the [`DM`] owned by the `snes` in the `user_f` by using
    /// [`let dm = snes.try_get_dm().unwrap();`](SNES::try_get_dm()).
    ///
    /// # Example
    ///
    /// See example: `snes-ex3` for full code (at 
    /// [`examples/snes/src/ex3.rs`](https://gitlab.com/petsc/petsc-rs/-/blob/main/examples/snes/src/ex3.rs)).
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # use mpi::traits::*;
    /// # fn main() -> petsc_rs::Result<()> {
    /// # let petsc = Petsc::init_no_args()?;
    /// let check_tol = 1.0;
    ///
    /// let mut snes = petsc.snes_create()?;
    /// snes.set_from_options()?;
    ///
    /// snes.linesearch_set_post_check(|_ls, snes, x_last, _y, x, _y_mod, x_mod| { 
    ///     let it = snes.get_iteration_number()?;
    ///     if it > 0 {
    ///         petsc_println!(petsc.world(),
    ///             "Checking candidate step at iteration {} with tolerance {}",
    ///             it, check_tol)?;
    ///         let xa_last = x_last.view()?;
    ///         let mut xa = x.view_mut()?;
    ///
    /// # // We want the complex case to pass the doc-test but we dont want to
    /// # // put both versions in the doc-example, thus the `#`s.
    /// #         #[cfg(feature = "petsc-use-complex-unsafe")]
    /// #         xa_last.indexed_iter().map(|(pat, _)|  pat[0])
    /// #             .for_each(|i| {
    /// #                 let rdiff = if xa[i].norm() == 0.0 { 2.0*check_tol }
    /// #                     else { ((xa[i] - xa_last[i])/xa[i]).norm() };
    /// #                 if rdiff > check_tol {
    /// #                     xa[i] = 0.5*(xa[i] + xa_last[i]);
    /// #                     *x_mod = true;
    /// #                 }
    /// #             });
    /// #         #[cfg(not(feature = "petsc-use-complex-unsafe"))]
    ///         xa_last.indexed_iter().map(|(pat, _)|  pat[0])
    ///             .for_each(|i| {
    ///                 // Note, for complex numbers you would use `.norm()`
    ///                 let rdiff = if xa[i].abs() == 0.0 { 2.0*check_tol }
    ///                     else { ((xa[i] - xa_last[i])/xa[i]).abs() };
    ///                 if rdiff > check_tol {
    ///                     xa[i] = 0.5*(xa[i] + xa_last[i]);
    ///                     *x_mod = true;
    ///                 }
    ///             });
    ///     }
    ///     Ok(())
    /// })?;
    ///
    /// // We would then set the function and jacobian and call solve on the snes context
    /// # Ok(())
    /// # }
    /// ```
    pub fn linesearch_set_post_check<F>(&mut self, user_f: F) -> Result<()>
    where
        F: FnMut(&LineSearch<'a>, &SNES<'a, 'tl, '_>, &Vector<'a>, &mut Vector<'a>, &mut Vector<'a>, &mut bool, &mut bool) -> Result<()> + 'tl
    {
        if self.linesearch.is_none() {
            // This just sets the linesearch
            let _ = self.get_linesearch_or_create()?;
        }

        let closure_anchor = Box::new(user_f);

        let trampoline_data = Box::pin(SNESLineSearchPostCheckTrampolineData { 
            world: self.world, user_f: closure_anchor });
        let _ = self.linecheck_post_check_trampoline_data.take();

        unsafe extern "C" fn snes_linesearch_set_post_check_trampoline(ls_p: *mut petsc_raw::_p_LineSearch,
            x_p: *mut petsc_raw::_p_Vec, y_p: *mut petsc_raw::_p_Vec, w_p: *mut petsc_raw::_p_Vec,
            changed_y_p: *mut petsc_raw::PetscBool, changed_w_p: *mut petsc_raw::PetscBool,
            ctx: *mut ::std::os::raw::c_void) -> petsc_raw::PetscErrorCode
        {
            // SAFETY: read `snes_function_trampoline` safety
            let mut trampoline_data: Pin<&mut SNESLineSearchPostCheckTrampolineData> = std::mem::transmute(ctx);

            let ls = ManuallyDrop::new(LineSearch { world: trampoline_data.world, ls_p });
            let mut snes_p = MaybeUninit::uninit();
            let ierr = petsc_raw::SNESLineSearchGetSNES(ls_p, snes_p.as_mut_ptr());
            if ierr != 0 { let _ = chkerrq!(trampoline_data.world, ierr); return ierr; }
            let mut snes = ManuallyDrop::new(SNES::new(trampoline_data.world, snes_p.assume_init()));

            let mut dm_p = MaybeUninit::uninit();
            let ierr = petsc_raw::SNESGetDM(snes_p.assume_init(), dm_p.as_mut_ptr());
            if ierr != 0 { let _ = chkerrq!(trampoline_data.world, ierr); return ierr; }
            let mut dm = DM::new(trampoline_data.world, dm_p.assume_init());
            let ierr = DM::set_inner_values_for_readonly(&mut dm);
            if ierr != 0 { return ierr; }
            snes.dm = Some(dm); // Note, because snes is not dropped, snes.dm wont be either
            
            let x_vec = ManuallyDrop::new(Vector { world: trampoline_data.world, vec_p: x_p });
            let mut y_vec = ManuallyDrop::new(Vector { world: trampoline_data.world, vec_p: y_p });
            let mut w_vec = ManuallyDrop::new(Vector { world: trampoline_data.world, vec_p: w_p });
            let mut changed_y = false;
            let mut changed_w = false;
            let mut changed_y_red = false;
            let mut changed_w_red = false;
            
            let res = (trampoline_data.deref_mut().user_f)(&ls, &snes, &x_vec, &mut y_vec, &mut w_vec, &mut changed_y, &mut changed_w)
                .map_or_else(|err| err.kind as i32, |_| 0);

            // TODO: should we do this? The C API doesn't do this, it just returns an error,
            // but I dont think that is very helpful. It doesn't seem like there is a good
            // reason to not just do this all reduce.
            let lor = mpi::collective::SystemOperation::logical_or();
            trampoline_data.world.all_reduce_into(slice::from_ref(&changed_y), slice::from_mut(&mut changed_y_red), lor);
            trampoline_data.world.all_reduce_into(slice::from_ref(&changed_w), slice::from_mut(&mut changed_w_red), lor);

            *changed_y_p = changed_y_red.into();
            *changed_w_p = changed_w_red.into();

            res
        }

        let ierr = unsafe { petsc_raw::SNESLineSearchSetPostCheck(
            self.linesearch.as_ref().unwrap().ls_p, Some(snes_linesearch_set_post_check_trampoline), 
            std::mem::transmute(trampoline_data.as_ref())) };
        unsafe { chkerrq!(self.world, ierr) }?;
        
        self.linecheck_post_check_trampoline_data = Some(trampoline_data);

        Ok(())
    }

    /// Sets a user function that is called after the initial search direction has been computed but
    /// before the line search routine has been applied. Allows the user to adjust the result of
    /// (usually a linear solve) that determined the search direction.
    ///
    /// # Parameters 
    ///
    /// * `user_f` - function evaluation routine
    ///     * `ls` - The linesearch context 
    ///     * `snes` - The snes context 
    ///     * `x` - The last step
    ///     * `y` - The step direction
    ///     * `changed_y` *(output)* - Indicator if the direction `y` has been changed.
    ///
    /// # Note
    ///
    /// Note, when the `pre_check` function is called, `petsc-rs` will automatically apply
    /// a logical OR via an MPI all reduce on the `change_y` value. That is to say, if you
    /// set one of the `change_y` values to `true` on any process, it will be treated as if all
    /// processes returned `true`. This is different from the C API which requires all the
    /// values to be logically collective (the same for each process).
    ///
    /// You can access the [`DM`] owned by the `snes` in the `user_f` by using
    /// [`let dm = snes.try_get_dm().unwrap();`](SNES::try_get_dm()).
    pub fn linesearch_set_pre_check<F>(&mut self, user_f: F) -> Result<()>
    where
        F: FnMut(&LineSearch<'a>, &SNES<'a, 'tl, '_>, &Vector<'a>, &mut Vector<'a>, &mut bool) -> Result<()> + 'tl
    {
        if self.linesearch.is_none() {
            // This just sets the linesearch if it isn't already
            let _ = self.get_linesearch_or_create()?;
        }

        let closure_anchor = Box::new(user_f);

        let trampoline_data = Box::pin(SNESLineSearchPreCheckTrampolineData { 
            world: self.world, user_f: closure_anchor });
        let _ = self.linecheck_pre_check_trampoline_data.take();

        unsafe extern "C" fn snes_linesearch_set_pre_check_trampoline(ls_p: *mut petsc_raw::_p_LineSearch,
            x_p: *mut petsc_raw::_p_Vec, y_p: *mut petsc_raw::_p_Vec,
            changed_y_p: *mut petsc_raw::PetscBool,
            ctx: *mut ::std::os::raw::c_void) -> petsc_raw::PetscErrorCode
        {
            // SAFETY: read `snes_function_trampoline` safety
            let mut trampoline_data: Pin<&mut SNESLineSearchPreCheckTrampolineData> = std::mem::transmute(ctx);

            let ls = ManuallyDrop::new(LineSearch { world: trampoline_data.world, ls_p });
            let mut snes_p = MaybeUninit::uninit();
            let ierr = petsc_raw::SNESLineSearchGetSNES(ls_p, snes_p.as_mut_ptr());
            if ierr != 0 { let _ = chkerrq!(trampoline_data.world, ierr); return ierr; }
            let mut snes = ManuallyDrop::new(SNES::new(trampoline_data.world, snes_p.assume_init()));

            let mut dm_p = MaybeUninit::uninit();
            let ierr = petsc_raw::SNESGetDM(snes_p.assume_init(), dm_p.as_mut_ptr());
            if ierr != 0 { let _ = chkerrq!(trampoline_data.world, ierr); return ierr; }
            let mut dm = DM::new(trampoline_data.world, dm_p.assume_init());
            let ierr = DM::set_inner_values_for_readonly(&mut dm);
            if ierr != 0 { return ierr; }
            snes.dm = Some(dm); // Note, because snes is not dropped, snes.dm wont be either
            
            let x_vec = ManuallyDrop::new(Vector { world: trampoline_data.world, vec_p: x_p });
            let mut y_vec = ManuallyDrop::new(Vector { world: trampoline_data.world, vec_p: y_p });
            let mut changed_y = false;
            let mut changed_y_red = false;
            
            let res = (trampoline_data.deref_mut().user_f)(&ls, &snes, &x_vec, &mut y_vec, &mut changed_y)
                .map_or_else(|err| err.kind as i32, |_| 0);

            // TODO: should we do this? The C API doesn't do this, it just returns an error,
            // but I dont think that is very helpful. It doesn't seem like there is a good
            // reason to not just do this all reduce.
            let lor = mpi::collective::SystemOperation::logical_or();
            trampoline_data.world.all_reduce_into(slice::from_ref(&changed_y), slice::from_mut(&mut changed_y_red), lor);

            *changed_y_p = changed_y_red.into();

            res
        }

        let ierr = unsafe { petsc_raw::SNESLineSearchSetPreCheck(
            self.linesearch.as_ref().unwrap().ls_p, Some(snes_linesearch_set_pre_check_trampoline), 
            std::mem::transmute(trampoline_data.as_ref())) };
        unsafe { chkerrq!(self.world, ierr) }?;
        
        self.linecheck_pre_check_trampoline_data = Some(trampoline_data);

        Ok(())
    }

    /// Solves a nonlinear system F(x) = b.
    ///
    /// # Parameters
    ///
    /// * `b` - the constant part of the equation F(x) = b, or None to use zero.
    /// * `x` - the solution vector
    ///
    /// The user should initialize the vector, `x`, with the initial guess for the nonlinear solve prior
    /// to calling [`SNES::solve()`]. In particular, to employ an initial guess of zero, the user should
    /// explicitly set this vector to zero by calling [`Vector::set_all()`].
    // TODO: should this take mut self
    pub fn solve<'vl, 'val: 'vl>(&mut self, b: impl Into<Option<&'vl Vector<'val>>>, x: &mut Vector) -> Result<()> {
        let ierr = unsafe { petsc_raw::SNESSolve(self.snes_p, b.into().map_or(std::ptr::null_mut(), |v| v.vec_p), x.vec_p) };
        unsafe { chkerrq!(self.world, ierr) }
    }

    /// Gets the [`SNES`] method type and name (as a [`String`]). 
    pub fn get_type_str(&self) -> Result<String> {
        let mut s_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::SNESGetType(self.snes_p, s_p.as_mut_ptr()) };
        unsafe { chkerrq!(self.world, ierr) }?;
        
        let c_str: &CStr = unsafe { CStr::from_ptr(s_p.assume_init()) };
        let str_slice: &str = c_str.to_str().unwrap();
        Ok(str_slice.to_owned())
    }

    /// Gets the solution vector for the linear system to be solved.
    pub fn get_solution(&self) -> Result<crate::vector::BorrowVector<'a, '_>> {
        let mut vec_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::SNESGetSolution(self.snes_p, vec_p.as_mut_ptr()) };
        unsafe { chkerrq!(self.world, ierr) }?;

        let vec = ManuallyDrop::new(Vector { world: self.world, vec_p: unsafe { vec_p.assume_init() } });
        let bvec = crate::vector::BorrowVector::new(vec, None);

        Ok(bvec)
    }
    
    /// Sets the [DM](DM) that may be used by some nonlinear solvers or their underlying
    /// [preconditioners](crate::pc).
    pub fn set_dm(&mut self, dm: DM<'a, 'tl>) -> Result<()> {
        
        let ierr = unsafe { petsc_raw::SNESSetDM(self.snes_p, dm.dm_p) };
        unsafe { chkerrq!(self.world, ierr) }?;

        let _ = self.dm.take();
        self.dm = Some(dm);

        Ok(())
    }

    /// Returns an [`Option`] to a reference to the [DM](DM).
    ///
    /// If you want PETSc to set the [`DM`] you must call
    /// [`SNES::get_dm_or_create()`], otherwise you must call [`SNES::set_dm()`]
    /// for this to return a `Some`.
    ///
    /// Note, this does not return a [`Result`](crate::Result) because it can never
    /// fail, instead it will return `None`.
    pub fn try_get_dm(&self) -> Option<&DM<'a, 'tl>> {
        self.dm.as_ref()
    }

    /// Returns a reference to the [DM](DM).
    pub fn get_dm(&mut self) -> Result<&DM<'a, 'tl>> {
        if self.dm.is_some() {
            Ok(self.dm.as_ref().unwrap())
        } else {
            let mut dm_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::SNESGetDM(self.snes_p, dm_p.as_mut_ptr()) };
            unsafe { chkerrq!(self.world, ierr) }?;

            self.dm = Some(DM::new(self.world, unsafe { dm_p.assume_init() }));
            unsafe { self.dm.as_mut().unwrap().reference()?; }

            Ok(self.dm.as_ref().unwrap())
        }
    }

    /// Returns a mutable reference to the [DM](DM).
    pub fn get_dm_or_create(&mut self) -> Result<&mut DM<'a, 'tl>> {
        if self.dm.is_some() {
            Ok(self.dm.as_mut().unwrap())
        } else {
            let mut dm_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::SNESGetDM(self.snes_p, dm_p.as_mut_ptr()) };
            unsafe { chkerrq!(self.world, ierr) }?;

            self.dm = Some(DM::new(self.world, unsafe { dm_p.assume_init() }));
            unsafe { self.dm.as_mut().unwrap().reference()?; }

            Ok(self.dm.as_mut().unwrap())
        }
    }

    /// Use DMPlex's internal FEM routines to compute SNES boundary values, residual, and Jacobian.
    pub fn use_dm_plex_local_fem(&mut self) -> Result<()> {
        let dm = self.get_dm_or_create()?;
        let ierr = unsafe { petsc_raw::DMPlexSetSNESLocalFEM(dm.dm_p,
            std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut()) };
        unsafe { chkerrq!(self.world, ierr) }
    }

    /// Returns the vector where the function is stored (i.e. the residual) or creates one
    /// if one hasn't been created.
    ///
    /// This is the vector `r` set with [`SNES::set_function()`].
    pub fn get_residual_or_create(&mut self) -> Result<&mut Vector<'a>> {
        let mut vec_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::SNESGetFunction(self.snes_p, vec_p.as_mut_ptr(),
            std::ptr::null_mut(), std::ptr::null_mut()) };
        unsafe { chkerrq!(self.world, ierr) }?;
        let vec_p = unsafe { vec_p.assume_init() };

        if let Some(r_vec) = self.residual_vec.as_deref() {
            if r_vec.vec_p == vec_p {
                return Ok(self.residual_vec.as_deref_mut().unwrap());
            }
        }
        let _ = self.residual_vec.take();
        let mut vec = Vector { world: self.world, vec_p };
        unsafe { vec.reference()?; }
        self.residual_vec_owned = Some(vec);

        Ok(self.residual_vec_owned.as_mut().unwrap())
    }

    /// Returns the vector where the function is stored (i.e. the residual) if one exists
    /// and has been set.
    ///
    /// This is the vector `r` set with [`SNES::set_function()`].
    pub fn try_get_residual(&self) -> Option<&Vector<'a>> {
        if self.residual_vec.is_some() {
            self.residual_vec.as_deref()
        } else { 
            self.residual_vec_owned.as_ref()
        }
    }

    /// Calls the function that has been set with [`SNES::set_function()`].
    ///
    /// Note, this method is typically used within nonlinear solvers implementations,
    /// so users would not generally call this routine themselves.
    pub fn compute_function(&mut self, x: &Vector, y: &mut Vector) -> Result<()> {
        let ierr = unsafe { petsc_raw::SNESComputeFunction(self.snes_p, x.vec_p, y.vec_p) };
        unsafe { chkerrq!(self.world, ierr) }
    }

    /// Computes the Jacobian matrices that has been set with [`SNES::set_jacobian()`]
    /// or [`SNES::set_jacobian_single_mat()`].
    ///
    /// Note, this method is typically used within nonlinear solvers implementations,
    /// so users would not generally call this routine themselves.
    pub fn compute_jacobian<'ml, 'mal: 'ml, 'mtl: 'ml>(&mut self, x: &Vector, a_mat: &mut Mat, p_mat: impl Into<Option<&'ml mut Mat<'mal, 'mtl>>>) -> Result<()> {
        let ierr = unsafe { petsc_raw::SNESComputeJacobian(self.snes_p, x.vec_p, a_mat.mat_p,
            p_mat.into().map_or(a_mat.mat_p, |p| p.mat_p)) };
        unsafe { chkerrq!(self.world, ierr) }
    }

    /// Determines whether a PETSc [`SNES`] is of a particular type.
    pub fn type_compare(&self, type_kind: SNESType) -> Result<bool> {
        self.type_compare_str(&type_kind.to_string())
    }
}

// macro impls
impl<'a> SNES<'a, '_, '_> {
    wrap_simple_petsc_member_funcs! {
        SNESSetFromOptions, pub set_from_options, takes mut, #[doc = "Sets various SNES and KSP parameters from user options."];
        SNESSetUp, pub set_up, takes mut, #[doc = "Sets up the internal data structures for the later use of a nonlinear solver. This will be automatically called with [`SNES::solve()`]."];
        SNESGetIterationNumber, pub get_iteration_number, output PetscInt, it_num, #[doc = "Gets the number of nonlinear iterations completed at this time. (<https://petsc.org/release/docs/manualpages/SNES/SNESGetIterationNumber.html>)"];
        SNESGetTolerances, pub get_tolerances, output PetscReal, atol, output PetscReal, rtol, output PetscReal, stol, output PetscInt, maxit, output PetscInt, maxf, #[doc = "Gets various parameters used in convergence tests.\n\n\
            # Outputs (in order)\n\n\
            * `atol` - absolute convergence tolerance\n\
            * `rtol` - relative convergence tolerance\n\
            * `stol` - convergence tolerance in terms of the norm of the change in the solution between steps\n\
            * `maxit` - maximum number of iterations\n\
            * `maxf` - maximum number of function evaluations\n"];
        SNESGetConvergedReason, pub get_converged_reason, output SNESConvergedReason, conv_reas, #[doc = "Gets the reason the SNES iteration was stopped."];
        SNESSetErrorIfNotConverged, pub set_error_if_not_converged, input bool, flg, takes mut, #[doc = "Causes [`SNES::solve()`] to generate an error if the solver has not converged.\n\n\
            Or the database key `-snes_error_if_not_converged` can be used.\n\nNormally PETSc continues if a linear solver fails to converge, you can call [`SNES::get_converged_reason()`] after a [`SNES::solve()`] to determine if it has converged."];
        SNESSetJacobianDomainError, pub(crate) set_jacobian_domain_error, takes mut, #[doc = "Tells [`SNES`] that compute jacobian does not make sense any more.\n\n\
            For example there is a negative element transformation. You probably want to use [`DomainErr`](DomainOrPetscError::DomainErr) instead of this function."];
        SNESSetFunctionDomainError, pub(crate) set_function_domain_error, takes mut, #[doc = "Tells [`SNES`] that the input vector to your SNES Function is not in the functions domain.\n\n\
            For example, negative pressure. You probably want to use [`DomainErr`](DomainOrPetscError::DomainErr) instead of this function."];
    }
}

impl_petsc_object_traits! {
    SNES, snes_p, petsc_raw::_p_SNES, SNESView, SNESDestroy, '_, '_;
    LineSearch, ls_p, petsc_raw::_p_LineSearch, SNESLineSearchView, SNESLineSearchDestroy;
}
