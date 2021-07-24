//! The Scalable Nonlinear Equations Solvers (SNES) component provides an easy-to-use interface to
//! Newton-type, quasi-Newton, full approximation scheme (FAS) multigrid, and other methods for solving
//! systems of nonlinear equations. 
//!
//! SNES users can set various algorithmic options at runtime via the
//! options database (e.g., specifying a trust region method via -snes_type newtontr ). SNES internally
//! employs [KSP](crate::ksp) for the solution of its linear systems. SNES users can also set [`KSP`](KSP) options
//! directly in application codes by first extracting the [`KSP`](KSP) context from the [`SNES`](crate::snes::SNES) context via
//! [`SNES::get_ksp()`](#) and then directly calling various [`KSP`](KSP) (and [`PC`](crate::pc::PC)) routines
//! (e.g., [`PC::set_type()`](#)).
//!
//! PETSc C API docs: <https://petsc.org/release/docs/manualpages/SNES/index.html>

use std:: pin::Pin;
use std::mem::{MaybeUninit, ManuallyDrop};
use std::ffi::CStr;
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
    ksp::{KSP, },
    dm::{DM, },
};
use mpi::topology::UserCommunicator;
use mpi::traits::*;

/// Abstract PETSc object that manages all nonlinear solves
pub struct SNES<'a, 'tl> {
    world: &'a UserCommunicator,
    pub(crate) snes_p: *mut petsc_raw::_p_SNES,

    ksp: Option<KSP<'a, 'tl>>,
    linesearch: Option<LineSearch<'a>>,
    dm: Option<DM<'a, 'tl>>,

    function_trampoline_data: Option<Pin<Box<SNESFunctionTrampolineData<'a, 'tl>>>>,
    jacobian_trampoline_data: Option<SNESJacobianTrampolineData<'a, 'tl>>,

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
    /// This will not create a `PetscError` internally unless you spesify that there should be an 
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

struct SNESFunctionTrampolineData<'a, 'tl> {
    world: &'a UserCommunicator,
    // This field is only used for its ownership/lifetime.
    // The usage of the pointer/reference is all handled on the c side.
    // However, we might want to use it for something like `get_residuals()`
    _vec: Option<&'tl mut Vector<'a>>,
    user_f: Box<dyn FnMut(&SNES<'a, 'tl>, &Vector<'a>, &mut Vector<'a>) -> std::result::Result<(), DomainOrPetscError> + 'tl>,
    set_dm: bool,
}

enum SNESJacobianTrampolineData<'a, 'tl> {
    SingleMat(Pin<Box<SNESJacobianSingleTrampolineData<'a, 'tl>>>),
    DoubleMat(Pin<Box<SNESJacobianDoubleTrampolineData<'a, 'tl>>>),
}

struct SNESJacobianSingleTrampolineData<'a, 'tl> {
    world: &'a UserCommunicator,
    _ap_mat: Mat<'a>,
    user_f: Box<dyn FnMut(&SNES<'a, 'tl>, &Vector<'a>, &mut Mat<'a>) -> std::result::Result<(), DomainOrPetscError> + 'tl>,
    set_dm: bool,
}

struct SNESJacobianDoubleTrampolineData<'a, 'tl> {
    world: &'a UserCommunicator,
    _a_mat: Mat<'a>,
    _p_mat: Mat<'a>,
    user_f: Box<dyn FnMut(&SNES<'a, 'tl>, &Vector<'a>, &mut Mat<'a>, &mut Mat<'a>) -> std::result::Result<(), DomainOrPetscError> + 'tl>,
    set_dm: bool,
}

struct SNESMonitorTrampolineData<'a, 'tl> {
    world: &'a UserCommunicator,
    user_f: Box<dyn FnMut(&SNES<'a, 'tl>, PetscInt, PetscReal) -> Result<()> + 'tl>,
    // set_dm: bool, // TODO: should we add this
}

struct SNESLineSearchPostCheckTrampolineData<'a, 'tl> {
    world: &'a UserCommunicator,
    user_f: Box<dyn FnMut(&LineSearch<'a>, &SNES<'a, 'tl>, &Vector<'a>, &mut Vector<'a>, &mut Vector<'a>, &mut bool, &mut bool) -> Result<()> + 'tl>,
    set_dm: bool,
}

struct SNESLineSearchPreCheckTrampolineData<'a, 'tl> {
    world: &'a UserCommunicator,
    user_f: Box<dyn FnMut(&LineSearch<'a>, &SNES<'a, 'tl>, &Vector<'a>, &mut Vector<'a>, &mut bool) -> Result<()> + 'tl>,
    set_dm: bool,
}

pub use petsc_raw::SNESConvergedReason;

impl<'a> Drop for SNES<'a, '_> {
    fn drop(&mut self) {
        let ierr = unsafe { petsc_raw::SNESDestroy(&mut self.snes_p as *mut _) };
        let _ = Petsc::check_error(self.world, ierr); // TODO: should I unwrap or what idk?
    }
}

impl Drop for LineSearch<'_> {
    fn drop(&mut self) {
        let ierr = unsafe { petsc_raw::SNESLineSearchDestroy(&mut self.ls_p as *mut _) };
        let _ = Petsc::check_error(self.world, ierr); // TODO: should I unwrap or what idk?
    }
}

impl<'a, 'tl> SNES<'a, 'tl> {
    /// Same as `SNES { ... }` but sets all optional params to `None`
    pub(crate) fn new(world: &'a UserCommunicator, snes_p: *mut petsc_raw::_p_SNES) -> Self {
        SNES { world, snes_p, ksp: None, function_trampoline_data: None,
               jacobian_trampoline_data: None, monitor_tramoline_data: None,
               linesearch: None, dm: None,
               linecheck_post_check_trampoline_data: None,
               linecheck_pre_check_trampoline_data: None, }
    }

    /// Creates a nonlinear solver context.
    pub fn create(world: &'a UserCommunicator) -> Result<Self> {
        let mut snes_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::SNESCreate(world.as_raw(), snes_p.as_mut_ptr()) };
        Petsc::check_error(world, ierr)?;

        Ok(SNES::new(world, unsafe { snes_p.assume_init() }))
    }

    /// Sets a [`KSP`](KSP) context for the SNES object to use.
    ///
    /// if you change the ksp by calling set again, then the original will be dropped.
    pub fn set_ksp(&mut self, ksp: KSP<'a, 'tl>) -> Result<()>
    {
        
        let ierr = unsafe { petsc_raw::SNESSetKSP(self.snes_p, ksp.ksp_p) };
        Petsc::check_error(self.world, ierr)?;

        let _ = self.ksp.take();
        self.ksp = Some(ksp);

        Ok(())
    }

    /// Returns an [`Option`] to a reference to the [`KSP`](KSP) context.
    ///
    /// If you want PETSc to set the [`KSP`] you must call [`SNES::get_ksp()`]
    /// or [`SNES::get_ksp_mut()`].
    ///
    /// Note, this does not return a [`Result`](crate::Result) because it can never
    /// fail, instead it will return `None`.
    pub fn try_get_ksp(&self) -> Option<&KSP<'a, 'tl>> {
        self.ksp.as_ref()
    }

    /// Returns a reference to the [`KSP`](KSP) context.
    pub fn get_ksp(&mut self) -> Result<&KSP<'a, 'tl>>
    {
        // TODO: should we even have a non mut one (or only `get_ksp_mut`)

        if let Some(ref ksp) = self.ksp
        {
            Ok(ksp)
        }
        else
        {
            let mut ksp_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::SNESGetKSP(self.snes_p, ksp_p.as_mut_ptr()) };
            Petsc::check_error(self.world, ierr)?;

            // It is now ok to drop this because we incremented the C reference counter
            self.ksp = Some(KSP::new(self.world, unsafe { ksp_p.assume_init() }));
            unsafe { self.ksp.as_mut().unwrap().reference()?; }
            
            Ok(self.ksp.as_ref().unwrap())
        }
    }

    /// Returns a mutable reference to the [`KSP`](KSP) context.
    pub fn get_ksp_mut(&mut self) -> Result<&mut KSP<'a, 'tl>>
    {
        if let Some(ref mut ksp) = self.ksp
        {
            Ok(ksp)
        }
        else
        {
            let mut ksp_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::SNESGetKSP(self.snes_p, ksp_p.as_mut_ptr()) };
            Petsc::check_error(self.world, ierr)?;

            // It is now ok to drop this because we incremented the C reference counter
            self.ksp = Some(KSP::new(self.world, unsafe { ksp_p.assume_init() }));
            unsafe { self.ksp.as_mut().unwrap().reference()?; }

            Ok(self.ksp.as_mut().unwrap())
        }
    }

    /// Sets the function evaluation routine and function vector for use by the SNES routines in solving
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
    /// [`let dm = snes.try_get_dm().unwrap();`](SNES::try_get_dm()). Note, this will only work
    /// if you set the dm with [`SNES::set_dm()`] BEFORE you call the
    /// [`set_function()`](SNES::set_function) method.
    ///
    /// # Example
    ///
    /// See example: snes ex2.rs for full code (at `examples/snes/src/ex2.rs`).
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # use mpi::traits::*;
    /// # fn main() -> petsc_rs::Result<()> {
    /// # let petsc = Petsc::init_no_args()?;
    /// let n = 10;
    /// let mut r = petsc.vec_create()?;
    /// r.set_sizes(None, n)?;
    /// r.set_from_options()?;
    /// let g = r.duplicate()?;
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
    /// // We would then set the jacobian and call solve of the snes context
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_function<'av: 'tl, F>(&mut self, input_vec: impl Into<Option<&'tl mut Vector<'av>>>, user_f: F) -> Result<()>
    where
        F: FnMut(&SNES<'a, 'tl>, &Vector<'a>, &mut Vector<'a>) -> std::result::Result<(), DomainOrPetscError> + 'tl
    {
        // TODO: should input_vec be consumed or passed by mut ref? We want to take mutable access for
        // it until the SNES is dropped. so either way its not like we can return mutable access to the
        // caller anyways. Or what if it is an `Rc<RefCell<Vector>>`, then we could remove the reference
        // at runtime

        // TODO: look at how rsmpi did the trampoline stuff:
        // https://github.com/rsmpi/rsmpi/blob/master/src/collective.rs#L1684
        // They used libffi, that could be a safer way to do it.
        
        let input_vec = input_vec.into();

        let closure_anchor = Box::new(user_f);

        let input_vec_p = input_vec.as_ref().map_or(std::ptr::null_mut(), |v| v.vec_p);
        // Note, we only store input_vec in the trampoline data so it isn't dropped,
        // we never actually use it.
        // TODO: is this transmute safe? I don't think it is? With out it, i have no idea how to
        // make this function work. The alternative and "correct" was to do this is to make `'av: 'a`,
        // or get rid of `'av` in place of `'a`. But, this doesn't work, and i dont know why.
        // I think using a transmute should be fine because we never edit anything todo with the world,
        // which is what has the `'a`/`'av` attached to it. We only touch what the vec_p points to 
        // (through the C API). In fact, in rust, we never touch `_vec` at all. It acts more as phantom
        // data than anything. Also, when we give the user access to the vec, it has the world from the
        // `trampoline_data`, i.e. from self.
        let trampoline_data = Box::pin(SNESFunctionTrampolineData { 
            world: self.world, _vec: unsafe { std::mem::transmute(input_vec) }, user_f: closure_anchor,
            set_dm: self.dm.is_some() });

        // drop old trampoline_data
        let _ = self.function_trampoline_data.take();

        unsafe extern "C" fn snes_function_trampoline(snes_p: *mut petsc_raw::_p_SNES, x_p: *mut petsc_raw::_p_Vec,
            f_p: *mut petsc_raw::_p_Vec, ctx: *mut std::os::raw::c_void) -> petsc_raw::PetscErrorCode
        {
            // SAFETY: We construct ctx to be a Pin<Box<SNESFunctionTrampolineData>> but pass it in as a *void
            // Box<T> is equivalent to *T (or &T) for ffi. Because the SNES owns the closure we can make sure
            // everything in it (and the closure its self) lives for at least as long as this function can be
            // called.
            // We don't construct a Box<> because we dont want to drop anything
            let trampoline_data: Pin<&mut SNESFunctionTrampolineData> = std::mem::transmute(ctx);

            // We don't want to drop anything, we are just using this to turn pointers 
            // of the underlining types (i.e. *mut petsc_raw::_p_SNES) into references
            // of the rust wrapper types.
            // Note, SNES has optional members that might have to be dropped, but because
            // we only give immutable access to the user_f we don't have to worry about that
            // as they will all stay `None`.
            // If `Vector` ever has optional parameters, they MUST be dropped manually.
            // SAFETY: even though snes is mut and thus we can set optional parameters, we don't
            // as we dont expose the mut to the user closure, we only use it with `set_jacobian_domain_error`
            let mut snes = ManuallyDrop::new(SNES::new(trampoline_data.world, snes_p));
            if trampoline_data.set_dm {
                let mut dm_p = MaybeUninit::uninit();
                let ierr = petsc_raw::SNESGetDM(snes_p, dm_p.as_mut_ptr());
                if ierr != 0 { let _ = Petsc::check_error(trampoline_data.world, ierr); return ierr; }
                let mut dm = DM::new(trampoline_data.world, dm_p.assume_init());
                let ierr = DM::set_inner_values(&mut dm);
                if ierr != 0 { return ierr; }
                snes.dm = Some(dm); // Note, because snes is not dropped, snes.dm wont be either
            }
            let x = ManuallyDrop::new(Vector { world: trampoline_data.world, vec_p: x_p });
            let mut f = ManuallyDrop::new(Vector { world: trampoline_data.world, vec_p: f_p });
            
            // TODO: is this safe, can we move the data in user_f by calling it
            (trampoline_data.get_unchecked_mut().user_f)(&snes, &x, &mut f)
                .map_or_else(|err| match err {
                    DomainOrPetscError::DomainErr => {
                        // TODO: `set_function_domain_error` doesn't take mut, but i think it should (because it does
                        // change the snes behind the pointer). However, it isn't the end of the world because, in the 
                        // rust api, there is no way for the closure to access this value that is set by this function.
                        // Or do we need to do something to account for interior mutability
                        let perr = snes.set_function_domain_error();
                        match perr {
                            Ok(_) => 0,
                            Err(perr) => perr.kind as i32
                        }
                    },
                    DomainOrPetscError::PetscErr(perr) => perr.kind as i32
                }, |_| 0)
        }

        let ierr = unsafe { petsc_raw::SNESSetFunction(
            self.snes_p, input_vec_p, Some(snes_function_trampoline), 
            std::mem::transmute(trampoline_data.as_ref())) }; // this will also erase the lifetimes
        Petsc::check_error(self.world, ierr)?;
        
        self.function_trampoline_data = Some(trampoline_data);

        Ok(())
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
    /// [`let dm = snes.try_get_dm().unwrap();`](SNES::try_get_dm()). Note, this will only work
    /// if you set the dm with [`SNES::set_dm()`] BEFORE you call the
    /// [`set_jacobian_single_mat()`](SNES::set_jacobian_single_mat) method.
    ///
    /// # Example
    ///
    /// See example: snes ex2.rs for full code (at `examples/snes/src/ex2.rs`).
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # use mpi::traits::*;
    /// # fn main() -> petsc_rs::Result<()> {
    /// # let petsc = Petsc::init_no_args()?;
    /// let n = 10;
    ///
    /// #[allow(non_snake_case)]
    /// let mut J = petsc.mat_create()?;
    /// J.set_sizes(None, None, Some(n), Some(n))?;
    /// J.set_from_options()?;
    /// J.seq_aij_set_preallocation(3, None)?;
    ///
    /// let mut snes = petsc.snes_create()?;
    ///
    /// snes.set_jacobian_single_mat(J,|_snes, x: &Vector, ap_mat: &mut Mat| {
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
    pub fn set_jacobian_single_mat<F>(&mut self, ap_mat: Mat<'a>, user_f: F) -> Result<()>
    where
        F: FnMut(&SNES<'a, 'tl>, &Vector<'a>, &mut Mat<'a>) -> std::result::Result<(), DomainOrPetscError> + 'tl,
    {
        // TODO: should we make ap_mat an `Rc<RefCell<Mat>>`

        let closure_anchor = Box::new(user_f);

        let aj_mat_p = ap_mat.mat_p;
        let trampoline_data = Box::pin(SNESJacobianSingleTrampolineData { 
            world: self.world, _ap_mat: ap_mat, user_f: closure_anchor,
            set_dm: self.dm.is_some() });
        let _ = self.jacobian_trampoline_data.take();

        unsafe extern "C" fn snes_jacobian_single_trampoline(snes_p: *mut petsc_raw::_p_SNES, vec_p: *mut petsc_raw::_p_Vec,
            mat1_p: *mut petsc_raw::_p_Mat, mat2_p: *mut petsc_raw::_p_Mat, ctx: *mut ::std::os::raw::c_void) -> petsc_raw::PetscErrorCode
        {
            // This assert should always be true based on how we constructed this wrapper
            assert_eq!(mat1_p, mat2_p);

            // SAFETY: read `snes_function_trampoline` safety
            let trampoline_data: Pin<&mut SNESJacobianSingleTrampolineData> = std::mem::transmute(ctx);

            // We don't want to drop anything, we are just using this to turn pointers 
            // of the underlining types (i.e. *mut petsc_raw::_p_SNES) into references
            // of the rust wrapper types.
            // If `Mat` ever has optional parameters, they MUST be dropped manually.
            // SAFETY: even though snes is mut and thus we can set optional parameters, we don't
            // as we dont expose the mut to the user closure, we only use it with `set_jacobian_domain_error`
            let mut snes = ManuallyDrop::new(SNES::new(trampoline_data.world, snes_p));
            if trampoline_data.set_dm {
                let mut dm_p = MaybeUninit::uninit();
                let ierr = petsc_raw::SNESGetDM(snes_p, dm_p.as_mut_ptr());
                if ierr != 0 { let _ = Petsc::check_error(trampoline_data.world, ierr); return ierr; }
                let mut dm = DM::new(trampoline_data.world, dm_p.assume_init());
                let ierr = DM::set_inner_values(&mut dm);
                if ierr != 0 { return ierr; }
                snes.dm = Some(dm); // Note, because snes is not dropped, snes.dm wont be either
            }
            let x = ManuallyDrop::new(Vector { world: trampoline_data.world, vec_p: vec_p });
            let mut a_mat = ManuallyDrop::new(Mat { world: trampoline_data.world, mat_p: mat1_p });
            
            (trampoline_data.get_unchecked_mut().user_f)(&snes, &x, &mut a_mat)
                .map_or_else(|err| match err {
                    DomainOrPetscError::DomainErr => {
                        // TODO: `set_jacobian_domain_error` doesn't take mut, but i think it should (because it does
                        // change the snes behind the pointer). However, it isn't the end of the world because, in the 
                        // rust api, there is no way for the closure to access this value that is set by this function.
                        // Or do we need to do something to account for interior mutability
                        let perr = snes.set_jacobian_domain_error();
                        match perr {
                            Ok(_) => 0,
                            Err(perr) => perr.kind as i32
                        }
                    },
                    DomainOrPetscError::PetscErr(perr) => perr.kind as i32
                }, |_| 0)
        }

        let ierr = unsafe { petsc_raw::SNESSetJacobian(
            self.snes_p, aj_mat_p, aj_mat_p, Some(snes_jacobian_single_trampoline), 
            std::mem::transmute(trampoline_data.as_ref())) };
        Petsc::check_error(self.world, ierr)?;
        
        self.jacobian_trampoline_data = Some(SNESJacobianTrampolineData::SingleMat(trampoline_data));

        Ok(())
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
    /// [`let dm = snes.try_get_dm().unwrap();`](SNES::try_get_dm()). Note, this will only work
    /// if you set the dm with [`SNES::set_dm()`] BEFORE you call the
    /// [`set_jacobian()`](SNES::set_jacobian) method.
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
    /// snes.set_jacobian(J, P,|_snes, x: &Vector, a_mat: &mut Mat, p_mat: &mut Mat| {
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
    pub fn set_jacobian<F>(&mut self, a_mat: Mat<'a>, p_mat: Mat<'a>, user_f: F) -> Result<()>
    where
        F: FnMut(&SNES<'a, 'tl>, &Vector<'a>, &mut Mat<'a>, &mut Mat<'a>) -> std::result::Result<(), DomainOrPetscError> + 'tl
    {
        // TODO: should we make a/p_mat an `Rc<RefCell<Mat>>`
        let closure_anchor = Box::new(user_f);

        let a_mat_p = a_mat.mat_p;
        let p_mat_p = p_mat.mat_p;
        let trampoline_data = Box::pin(SNESJacobianDoubleTrampolineData { 
            world: self.world, _a_mat: a_mat, _p_mat: p_mat, user_f: closure_anchor,
            set_dm: self.dm.is_some() });
        let _ = self.jacobian_trampoline_data.take();

        unsafe extern "C" fn snes_jacobian_double_trampoline(snes_p: *mut petsc_raw::_p_SNES, vec_p: *mut petsc_raw::_p_Vec,
            mat1_p: *mut petsc_raw::_p_Mat, mat2_p: *mut petsc_raw::_p_Mat, ctx: *mut ::std::os::raw::c_void) -> petsc_raw::PetscErrorCode
        {
            // This assert should always be true based on how we constructed this wrapper
            assert_ne!(mat1_p, mat2_p);

            // SAFETY: read `snes_function_trampoline` safety
            let trampoline_data: Pin<&mut SNESJacobianDoubleTrampolineData> = std::mem::transmute(ctx);

            // We don't want to drop anything, we are just using this to turn pointers 
            // of the underlining types (i.e. *mut petsc_raw::_p_SNES) into references.
            // If `Mat` ever has optional parameters, they MUST be dropped manually.
            // SAFETY: even though snes is mut and thus we can set optional parameters, we don't
            // as we dont expose the mut to the user closure, we only use it with `set_jacobian_domain_error`
            let mut snes = ManuallyDrop::new(SNES::new(trampoline_data.world, snes_p));
            if trampoline_data.set_dm {
                let mut dm_p = MaybeUninit::uninit();
                let ierr = petsc_raw::SNESGetDM(snes_p, dm_p.as_mut_ptr());
                if ierr != 0 { let _ = Petsc::check_error(trampoline_data.world, ierr); return ierr; }
                let mut dm = DM::new(trampoline_data.world, dm_p.assume_init());
                let ierr = DM::set_inner_values(&mut dm);
                if ierr != 0 { return ierr; }
                snes.dm = Some(dm); // Note, because snes is not dropped, snes.dm wont be either
            }
            let x = ManuallyDrop::new(Vector { world: trampoline_data.world, vec_p: vec_p });
            let mut a_mat = ManuallyDrop::new(Mat { world: trampoline_data.world, mat_p: mat1_p });
            let mut p_mat = ManuallyDrop::new(Mat { world: trampoline_data.world, mat_p: mat2_p });
            
            (trampoline_data.get_unchecked_mut().user_f)(&snes, &x, &mut a_mat, &mut p_mat)
                .map_or_else(|err| match err {
                    DomainOrPetscError::DomainErr => {
                        let perr = snes.set_jacobian_domain_error();
                        match perr {
                            Ok(_) => 0,
                            Err(perr) => perr.kind as i32
                        }
                    },
                    DomainOrPetscError::PetscErr(perr) => perr.kind as i32
                }, |_| 0)
        }

        let ierr = unsafe { petsc_raw::SNESSetJacobian(
            self.snes_p, a_mat_p, p_mat_p, Some(snes_jacobian_double_trampoline), 
            std::mem::transmute(trampoline_data.as_ref())) };
        Petsc::check_error(self.world, ierr)?;
        
        self.jacobian_trampoline_data = Some(SNESJacobianTrampolineData::DoubleMat(trampoline_data));

        Ok(())
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
        F: FnMut(&SNES<'a, 'tl>, PetscInt, PetscReal) -> Result<()> + 'tl
    {
        // TODO: should we make a/p_mat an `Rc<RefCell<Mat>>`
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
            
            (trampoline_data.get_unchecked_mut().user_f)(&snes, it, norm)
                .map_or_else(|err| err.kind as i32, |_| 0)
        }

        let ierr = unsafe { petsc_raw::SNESMonitorSet(
            self.snes_p, Some(snes_monitor_trampoline), 
            std::mem::transmute(trampoline_data.as_ref()), 
            None) }; // We dont need to tell C the drop function because rust will take care of it for us.
        Petsc::check_error(self.world, ierr)?;
        
        self.monitor_tramoline_data = Some(trampoline_data);

        Ok(())
    }

    /// Returns an immutable reference to the line search context set with `SNESSetLineSearch()`
    /// or creates a default line search instance associated with the SNES and returns it. 
    pub fn get_linesearch(&mut self) -> Result<&LineSearch<'a>> {
        if self.linesearch.is_some() {
            Ok(self.linesearch.as_ref().unwrap())
        } else {
            let mut ls_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::SNESGetLineSearch(self.snes_p, ls_p.as_mut_ptr()) };
            Petsc::check_error(self.world, ierr)?;

            self.linesearch = Some(LineSearch { world: self.world, ls_p: unsafe { ls_p.assume_init() } });
            unsafe { self.linesearch.as_mut().unwrap().reference()?; }

            Ok(self.linesearch.as_ref().unwrap())
        }
    }

    /// Returns a mutable reference to the line search context set with `SNESSetLineSearch()`
    /// or creates a default line search instance associated with the SNES and returns it. 
    pub fn get_linesearch_mut(&mut self) -> Result<&mut LineSearch<'a>> {
        if self.linesearch.is_some() {
            Ok(self.linesearch.as_mut().unwrap())
        } else {
            let mut ls_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::SNESGetLineSearch(self.snes_p, ls_p.as_mut_ptr()) };
            Petsc::check_error(self.world, ierr)?;

            self.linesearch = Some(LineSearch { world: self.world, ls_p: unsafe { ls_p.assume_init() } });
            unsafe { self.linesearch.as_mut().unwrap().reference()?; }

            Ok(self.linesearch.as_mut().unwrap())
        }
    }

    // TODO: we can do better that having the user say if they edited the vec (how does Cow work? can we use that here?)
    // Also note that if we are running on multiple processes, the changed flag must be the same for all processes.
    /// Sets a user function that is called after the line search has been applied to determine the step
    /// direction and length. Allows the user to change or override the decision of the line search routine.
    ///
    /// # Parameters 
    ///
    /// * `user_f` - function evaluation routine
    ///     * `ls` - The linesearch context 
    ///     * `snes` - The snes context 
    ///     * `x` - The last step
    ///     * `y` - The step direction 
    ///     * `w` - The updated solution, `w = x + lambda*y` for some lambda.
    ///     * `changed_y` - Indicator if the direction `y` has been changed.
    ///     * `changed_w` - Indicator if the new candidate solution `w` has been changed.
    ///
    /// # Note
    ///
    /// You can access the [`DM`] owned by the `snes` in the `user_f` by using
    /// [`let dm = snes.try_get_dm().unwrap();`](SNES::try_get_dm()). Note, this will only work
    /// if you set the dm with [`SNES::set_dm()`] BEFORE you call the
    /// [`linesearch_set_post_check()`](SNES::linesearch_set_post_check) method.
    pub fn linesearch_set_post_check<F>(&mut self, user_f: F) -> Result<()>
    where
        F: FnMut(&LineSearch<'a>, &SNES<'a, 'tl>, &Vector<'a>, &mut Vector<'a>, &mut Vector<'a>, &mut bool, &mut bool) -> Result<()> + 'tl
    {
        if self.linesearch.is_none() {
            // This just sets the linesearch
            let _ = self.get_linesearch()?;
        }

        let closure_anchor = Box::new(user_f);

        let trampoline_data = Box::pin(SNESLineSearchPostCheckTrampolineData { 
            world: self.world, user_f: closure_anchor, set_dm: self.dm.is_some() });
        let _ = self.linecheck_post_check_trampoline_data.take();

        unsafe extern "C" fn snes_linesearch_set_post_check_trampoline(ls_p: *mut petsc_raw::_p_LineSearch,
            x_p: *mut petsc_raw::_p_Vec, y_p: *mut petsc_raw::_p_Vec, w_p: *mut petsc_raw::_p_Vec,
            changed_y_p: *mut petsc_raw::PetscBool, changed_w_p: *mut petsc_raw::PetscBool,
            ctx: *mut ::std::os::raw::c_void) -> petsc_raw::PetscErrorCode
        {
            // SAFETY: read `snes_function_trampoline` safety
            let trampoline_data: Pin<&mut SNESLineSearchPostCheckTrampolineData> = std::mem::transmute(ctx);

            let ls = ManuallyDrop::new(LineSearch { world: trampoline_data.world, ls_p });
            let mut snes_p = MaybeUninit::uninit();
            let ierr = petsc_raw::SNESLineSearchGetSNES(ls_p, snes_p.as_mut_ptr());
            if ierr != 0 { let _ = Petsc::check_error(trampoline_data.world, ierr); return ierr; }
            let mut snes = ManuallyDrop::new(SNES::new(trampoline_data.world, snes_p.assume_init()));
            if trampoline_data.set_dm {
                let mut dm_p = MaybeUninit::uninit();
                let ierr = petsc_raw::SNESGetDM(snes_p.assume_init(), dm_p.as_mut_ptr());
                if ierr != 0 { let _ = Petsc::check_error(trampoline_data.world, ierr); return ierr; }
                let mut dm = DM::new(trampoline_data.world, dm_p.assume_init());
                let ierr = DM::set_inner_values(&mut dm);
                if ierr != 0 { return ierr; }
                snes.dm = Some(dm); // Note, because snes is not dropped, snes.dm wont be either
            }
            
            let x_vec = ManuallyDrop::new(Vector { world: trampoline_data.world, vec_p: x_p });
            let mut y_vec = ManuallyDrop::new(Vector { world: trampoline_data.world, vec_p: y_p });
            let mut w_vec = ManuallyDrop::new(Vector { world: trampoline_data.world, vec_p: w_p });
            let mut changed_y = false;
            let mut changed_w = false;
            
            let res = (trampoline_data.get_unchecked_mut().user_f)(&ls, &snes, &x_vec, &mut y_vec, &mut w_vec, &mut changed_y, &mut changed_w)
                .map_or_else(|err| err.kind as i32, |_| 0);

            *changed_y_p = changed_y.into();
            *changed_w_p = changed_w.into();

            res
        }

        let ierr = unsafe { petsc_raw::SNESLineSearchSetPostCheck(
            self.linesearch.as_ref().unwrap().ls_p, Some(snes_linesearch_set_post_check_trampoline), 
            std::mem::transmute(trampoline_data.as_ref())) };
        Petsc::check_error(self.world, ierr)?;
        
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
    ///     * `changed_y` - Indicator if the direction `y` has been changed.
    ///
    /// # Note
    ///
    /// You can access the [`DM`] owned by the `snes` in the `user_f` by using
    /// [`let dm = snes.try_get_dm().unwrap();`](SNES::try_get_dm()). Note, this will only work
    /// if you set the dm with [`SNES::set_dm()`] BEFORE you call the
    /// [`linesearch_set_pre_check()`](SNES::linesearch_set_pre_check) method.
    pub fn linesearch_set_pre_check<F>(&mut self, user_f: F) -> Result<()>
    where
        F: FnMut(&LineSearch<'a>, &SNES<'a, 'tl>, &Vector<'a>, &mut Vector<'a>, &mut bool) -> Result<()> + 'tl
    {
        if self.linesearch.is_none() {
            // This just sets the linesearch if it isn't already
            let _ = self.get_linesearch()?;
        }

        let closure_anchor = Box::new(user_f);

        let trampoline_data = Box::pin(SNESLineSearchPreCheckTrampolineData { 
            world: self.world, user_f: closure_anchor, set_dm: self.dm.is_some() });
        let _ = self.linecheck_pre_check_trampoline_data.take();

        unsafe extern "C" fn snes_linesearch_set_pre_check_trampoline(ls_p: *mut petsc_raw::_p_LineSearch,
            x_p: *mut petsc_raw::_p_Vec, y_p: *mut petsc_raw::_p_Vec,
            changed_y_p: *mut petsc_raw::PetscBool,
            ctx: *mut ::std::os::raw::c_void) -> petsc_raw::PetscErrorCode
        {
            // SAFETY: read `snes_function_trampoline` safety
            let trampoline_data: Pin<&mut SNESLineSearchPreCheckTrampolineData> = std::mem::transmute(ctx);

            let ls = ManuallyDrop::new(LineSearch { world: trampoline_data.world, ls_p });
            let mut snes_p = MaybeUninit::uninit();
            let ierr = petsc_raw::SNESLineSearchGetSNES(ls_p, snes_p.as_mut_ptr());
            if ierr != 0 { let _ = Petsc::check_error(trampoline_data.world, ierr); return ierr; }
            let mut snes = ManuallyDrop::new(SNES::new(trampoline_data.world, snes_p.assume_init()));
            if trampoline_data.set_dm {
                let mut dm_p = MaybeUninit::uninit();
                let ierr = petsc_raw::SNESGetDM(snes_p.assume_init(), dm_p.as_mut_ptr());
                if ierr != 0 { let _ = Petsc::check_error(trampoline_data.world, ierr); return ierr; }
                let mut dm = DM::new(trampoline_data.world, dm_p.assume_init());
                let ierr = DM::set_inner_values(&mut dm);
                if ierr != 0 { return ierr; }
                snes.dm = Some(dm); // Note, because snes is not dropped, snes.dm wont be either
            }
            
            let x_vec = ManuallyDrop::new(Vector { world: trampoline_data.world, vec_p: x_p });
            let mut y_vec = ManuallyDrop::new(Vector { world: trampoline_data.world, vec_p: y_p });
            let mut changed_y = false;
            
            let res = (trampoline_data.get_unchecked_mut().user_f)(&ls, &snes, &x_vec, &mut y_vec, &mut changed_y)
                .map_or_else(|err| err.kind as i32, |_| 0);

            *changed_y_p = changed_y.into();

            res
        }

        let ierr = unsafe { petsc_raw::SNESLineSearchSetPreCheck(
            self.linesearch.as_ref().unwrap().ls_p, Some(snes_linesearch_set_pre_check_trampoline), 
            std::mem::transmute(trampoline_data.as_ref())) };
        Petsc::check_error(self.world, ierr)?;
        
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
    pub fn solve(&self, b: Option<&Vector>, x: &mut Vector) -> Result<()>
    {
        let ierr = unsafe { petsc_raw::SNESSolve(self.snes_p, b.map_or(std::ptr::null_mut(), |v| v.vec_p), x.vec_p) };
        Petsc::check_error(self.world, ierr)
    }

    /// Gets the SNES method type and name (as a [`String`]). 
    pub fn get_type(&self) -> Result<String> {
        // TODO: return enum
        let mut s_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::SNESGetType(self.snes_p, s_p.as_mut_ptr()) };
        Petsc::check_error(self.world, ierr)?;
        
        let c_str: &CStr = unsafe { CStr::from_ptr(s_p.assume_init()) };
        let str_slice: &str = c_str.to_str().unwrap();
        Ok(str_slice.to_owned())
    }

    /// Gets the solution vector for the linear system to be solved.
    pub fn get_solution(&self) -> Result<Rc<Vector<'a>>> {
        let mut vec_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::SNESGetSolution(self.snes_p, vec_p.as_mut_ptr()) };
        Petsc::check_error(self.world, ierr)?;

        let mut vec = Vector { world: self.world, vec_p: unsafe { vec_p.assume_init() } };
        unsafe { vec.reference()?; }
        let rhs = Rc::new(vec);

        Ok(rhs)
    }
    
    /// Sets the [DM](DM) that may be used by some nonlinear solvers or their underlying
    /// [preconditioners](crate::pc).
    pub fn set_dm(&mut self, dm: DM<'a, 'tl>) -> Result<()> {
        
        let ierr = unsafe { petsc_raw::SNESSetDM(self.snes_p, dm.dm_p) };
        Petsc::check_error(self.world, ierr)?;

        let _ = self.dm.take();
        self.dm = Some(dm);

        Ok(())
    }
    
    // TODO: should we add any of the following
    // // TODO: would it be safe to clone the dm and give the clone to the snes?
    // pub fn set_dm_from_mut(&mut self, dm: &mut DM<'a>) -> Result<()> {
    //
    //     let ierr = unsafe { petsc_raw::SNESSetDM(self.snes_p, dm.dm_p) };
    //     Petsc::check_error(self.world, ierr)?;
    //
    //     let dm_owned = dm.clone();
    //     let ierr = unsafe { petsc_raw::SNESSetDM(self.snes_p, dm_owned.dm_p) };
    //     Petsc::check_error(self.world, ierr)?;
    //
    //     let _ = self.dm.take();
    //     self.dm = Some(dm_owned);
    //
    //     Ok(())
    // }
    //
    // // TODO: omg this is not safe, even for unsafe
    // pub unsafe fn set_dm_from_ref(&mut self, dm: &DM<'a>) -> Result<()> {
    //     let ierr = petsc_raw::SNESSetDM(self.snes_p, dm.dm_p);
    //     Petsc::check_error(self.world, ierr)?;
    //
    //     let dm_owned = DM::new(dm.world, dm.dm_p);
    //     let ierr = petsc_raw::PetscObjectReference(dm_owned.dm_p as *mut petsc_raw::_p_PetscObject);
    //     Petsc::check_error(self.world, ierr)?;
    //
    //     let _ = self.dm.take();
    //     self.dm = Some(dm_owned);
    //
    //     Ok(())
    // }
    //
    // // TODO: this is not safe
    // pub fn set_dm_from_mut2(&mut self, dm: &mut DM<'a>) -> Result<()> {
    //     let ierr = unsafe { petsc_raw::SNESSetDM(self.snes_p, dm.dm_p) };
    //     Petsc::check_error(self.world, ierr)?;
    //
    //     let dm_owned = DM::new(dm.world, dm.dm_p);
    //     let ierr = unsafe { petsc_raw::PetscObjectReference(dm_owned.dm_p as *mut petsc_raw::_p_PetscObject) };
    //     Petsc::check_error(self.world, ierr)?;
    //
    //     let _ = self.dm.take();
    //     self.dm = Some(dm_owned);
    //
    //     Ok(())
    // }

    /// Returns an [`Option`] to a reference to the [DM](DM).
    ///
    /// If you want PETSc to set the [`DM`] you must call [`SNES::get_dm()`]
    /// or [`SNES::get_dm_mut()`], otherwise you must call [`SNES::set_dm()`]
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
            Petsc::check_error(self.world, ierr)?;

            self.dm = Some(DM::new(self.world, unsafe { dm_p.assume_init() }));
            unsafe { self.dm.as_mut().unwrap().reference()?; }

            Ok(self.dm.as_ref().unwrap())
        }
    }

    /// Returns a mutable reference to the [DM](DM).
    pub fn get_dm_mut(&mut self) -> Result<&mut DM<'a, 'tl>> {
        if self.dm.is_some() {
            Ok(self.dm.as_mut().unwrap())
        } else {
            let mut dm_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::SNESGetDM(self.snes_p, dm_p.as_mut_ptr()) };
            Petsc::check_error(self.world, ierr)?;

            self.dm = Some(DM::new(self.world, unsafe { dm_p.assume_init() }));
            unsafe { self.dm.as_mut().unwrap().reference()?; }

            Ok(self.dm.as_mut().unwrap())
        }
    }

    // TODO: should this be here or in DM
    /// Use DMPlex's internal FEM routines to compute SNES boundary values, residual, and Jacobian.
    pub fn dm_plex_local_fem(&mut self) -> Result<()> {
        let dm = self.get_dm_mut()?;
        let ierr = unsafe { petsc_raw::DMPlexSetSNESLocalFEM(dm.dm_p,
            std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut()) };
        Petsc::check_error(self.world, ierr)
    }
}

// macro impls
impl<'a> SNES<'a, '_> {
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
        // TODO: these use interior mutability without UnsafeCell, is that ok? do we need to use UnsafeCell?
        SNESSetJacobianDomainError, pub(crate) set_jacobian_domain_error, takes mut, #[doc = "Tells [`SNES`] that compute jacobian does not make sense any more.\n\n\
            For example there is a negative element transformation. You probably want to use [`DomainErr`] instead of this function."];
        SNESSetFunctionDomainError, pub(crate) set_function_domain_error, takes mut, #[doc = "Tells [`SNES`] that the input vector to your SNES Function is not in the functions domain.\n\n\
            For example, negative pressure. You probably want to use [`DomainErr`] instead of this function."];
    }
}

impl_petsc_object_traits! { SNES, snes_p, petsc_raw::_p_SNES, '_ }

impl_petsc_view_func!{ SNES, SNESView, '_ }

impl_petsc_object_traits! { LineSearch, ls_p, petsc_raw::_p_LineSearch }

impl_petsc_view_func!{ LineSearch, SNESLineSearchView }
