//! The Scalable Nonlinear Equations Solvers (SNES) component provides an easy-to-use interface to
//! Newton-type, quasi-Newton, full approximation scheme (FAS) multigrid, and other methods for solving
//! systems of nonlinear equations. 
//!
//! SNES users can set various algorithmic options at runtime via the
//! options database (e.g., specifying a trust region method via -snes_type newtontr ). SNES internally
//! employs [KSP](crate::ksp) for the solution of its linear systems. SNES users can also set [`KSP`](KSP) options
//! directly in application codes by first extracting the [`KSP`](KSP) context from the [`SNES`](SNES) context via
//! [`SNES::get_ksp()`](#) and then directly calling various [`KSP`](KSP) (and [`PC`](PC)) routines
//! (e.g., [`PC::set_type()`](#)).
//!
//! PETSc C API docs: <https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/SNES/index.html>

use std::{mem::ManuallyDrop, pin::Pin};

use crate::prelude::*;

/// Abstract PETSc object that manages all nonlinear solves
pub struct SNES<'a, 'tl> {
    world: &'a dyn Communicator,
    pub(crate) snes_p: *mut petsc_raw::_p_SNES,

    #[doc(hidden)]
    ksp: Option<KSP<'a>>,

    function_trampoline_data: Option<Pin<Box<SNESFunctionTrampolineData<'a, 'tl>>>>,
    jacobian_single_trampoline_data: Option<Pin<Box<SNESJacobianSingleTrampolineData<'a, 'tl>>>>,
    jacobian_double_trampoline_data: Option<Pin<Box<SNESJacobianDoubleTrampolineData<'a, 'tl>>>>,
}

//pub type SNESFunctionUserClosure = dyn FnMut(&SNES, &Vector, &mut Vector) -> Result<()>;

struct SNESFunctionTrampolineData<'a, 'tl> {
    // TODO: idk what to put here
    world: &'a dyn Communicator,
    // This field is only used for its ownership.
    // The usage of the pointer/reference is all handled on the c side.
    // However, we might want to use it for something like `get_residuals()`
    _vec: Vector<'a>,
    user_f: Box<dyn FnMut(&SNES, &Vector, &mut Vector) -> Result<()> + 'tl>,
}

struct SNESJacobianSingleTrampolineData<'a, 'tl> {
    // TODO: idk what to put here
    world: &'a dyn Communicator,
    _ap_mat: Mat<'a>,
    user_f: Box<dyn FnMut(&SNES, &Vector, &mut Mat) -> Result<()> + 'tl>,
}

struct SNESJacobianDoubleTrampolineData<'a, 'tl> {
    // TODO: idk what to put here
    world: &'a dyn Communicator,
    _a_mat: Mat<'a>,
    _p_mat: Mat<'a>,
    user_f: Box<dyn FnMut(&SNES, &Vector, &mut Mat, &mut Mat) -> Result<()> + 'tl>,
}

pub use petsc_raw::SNESConvergedReason;

impl<'a> Drop for SNES<'a, '_> {
    fn drop(&mut self) {
        unsafe {
            let ierr = petsc_raw::SNESDestroy(&mut self.snes_p as *mut *mut petsc_raw::_p_SNES);
            let _ = Petsc::check_error(self.world, ierr); // TODO: should I unwrap or what idk?
        }
    }
}

impl<'a, 'b, 'tl> SNES<'a, 'tl> {
    /// Same as `SNES { ... }` but sets all optional params to `None`
    pub(crate) fn new(world: &'a dyn Communicator, snes_p: *mut petsc_raw::_p_SNES) -> Self {
        SNES { world, snes_p, ksp: None, function_trampoline_data: None,
             jacobian_single_trampoline_data: None, jacobian_double_trampoline_data: None }
    }

    /// Creates a nonlinear solver context.
    pub fn create(world: &'a dyn Communicator) -> Result<Self> {
        let mut snes_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::SNESCreate(world.as_raw(), snes_p.as_mut_ptr()) };
        Petsc::check_error(world, ierr)?;

        Ok(SNES::new(world, unsafe { snes_p.assume_init() }))
    }

    /// Sets a [`KSP`](KSP) context for the SNES object to use.
    ///
    /// if you change the ksp by calling set again, then the original will be dropped.
    pub fn set_ksp(&mut self, ksp: KSP<'a>) -> Result<()>
    {
        
        let ierr = unsafe { petsc_raw::SNESSetKSP(self.snes_p, ksp.ksp_p) };
        Petsc::check_error(self.world, ierr)?;

        let _ = self.ksp.take();
        self.ksp = Some(ksp);

        Ok(())
    }

    /// Returns a reference to the [`KSP`](KSP) context.
    pub fn get_ksp<'c>(&'c mut self) -> Result<&'c KSP<'a>>
    {
        // TODO: should we even have a non mut one (or only `get_ksp_mut`)

        // Under the hood, if the ksp is already set, i.e. with `set_ksp`, then `SNESGetKSP` just returns a pointer
        // to that, so we can bypass calling SNESGetKSP. However, there shouldn't be any problem with just calling
        // `SNESGetKSP` again as we incremented the reference of the KSP we "own" so dropping it wont do anything.
        if let Some(ref ksp) = self.ksp
        {
            Ok(ksp)
        }
        else
        {
            let mut ksp_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::SNESGetKSP(self.snes_p, ksp_p.as_mut_ptr()) };
            Petsc::check_error(self.world, ierr)?;
            let ierr = unsafe { petsc_raw::PetscObjectReference(ksp_p.assume_init() as *mut petsc_raw::_p_PetscObject) };
            Petsc::check_error(self.world, ierr)?;

            self.ksp = Some(KSP::new(self.world, unsafe { ksp_p.assume_init() }));

            Ok(self.ksp.as_ref().unwrap())
        }
    }

    /// Returns a mutable reference to the [`KSP`](KSP) context.
    pub fn get_ksp_mut<'c>(&'c mut self) -> Result<&'c mut KSP<'a>>
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
            let ierr = unsafe { petsc_raw::PetscObjectReference(ksp_p.assume_init() as *mut petsc_raw::_p_PetscObject) };
            Petsc::check_error(self.world, ierr)?;

            self.ksp = Some(KSP::new(self.world, unsafe { ksp_p.assume_init() }));

            Ok(self.ksp.as_mut().unwrap())
        }
    }

    /// Sets the function evaluation routine and function vector for use by the SNES routines in solving
    /// systems of nonlinear equations.
    ///
    /// # Parameters
    ///
    /// * `input_vec` - vector to store function value
    /// * `user_f` - A closure used to convey the nonlinear function to be solved by SNES
    ///     * `snes` - the snes context
    ///     * `x` - state at which to evaluate residual
    ///     * `f` - vector to put residual (function value)
    ///
    /// # Note
    ///
    /// The Newton-like methods typically solve linear systems of the form
    /// ```text
    /// f'(x) x = -f(x),
    /// ```
    /// where `f'(x)` denotes the Jacobian matrix and `f(x)` is the function.
    ///
    /// # Example
    ///
    /// See example: snes ex2.rs for full code (at `examples/snes/src/ex2.rs`).
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # fn main() -> petsc_rs::Result<()> {
    /// # let petsc = Petsc::init_no_args()?;
    /// let n = 10;
    /// let mut r = petsc.vec_create()?;
    /// r.set_sizes(None, Some(n))?;
    /// r.set_from_options()?;
    /// let mut g = r.duplicate()?;
    ///
    /// let mut snes = petsc.snes_create()?;
    ///
    /// snes.set_function(r, |_snes, x: &Vector, f: &mut Vector| {
    ///     let x_view = x.view()?;
    ///     let mut f_view = f.view_mut()?;
    ///     let g_view = g.view()?;
    ///
    ///     let d = f64::powi(n as f64 - 1.0, 2);
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
    pub fn set_function<F>(&mut self, input_vec: Vector<'a>, user_f: F) -> Result<()>
    where
        F: FnMut(&SNES, &Vector, &mut Vector) -> Result<()> + 'tl
    {
        // TODO: should input_vec be consumed or passed by mut ref? We want to take mutable access for
        // it until the SNES is dropped. so either way its not like we can return mutable to the caller
        // access anyways. Or what if it is an `Rc<RefCell<Vector>>`

        // Note, we give `SNESSetFunction` a pointer so that the SNES can edit the vec. That is why we also 
        // store a reference (that we never use). So we use the rust borrow checker.

        // TODO: look at how rsmpi did it: https://github.com/rsmpi/rsmpi/blob/master/src/collective.rs#L1684
        // They used libffi

        let anchor = Box::new(user_f);

        let input_vec_p = input_vec.vec_p;
        let trampoline_data = unsafe {
            Pin::new_unchecked(Box::new(SNESFunctionTrampolineData { 
                world: self.world, _vec: input_vec, user_f: anchor }))};
        let _ = self.function_trampoline_data.take();

        unsafe extern "C" fn snes_function_trampoline(snes_p: *mut petsc_raw::_p_SNES, x_p: *mut petsc_raw::_p_Vec,
            f_p: *mut petsc_raw::_p_Vec, ctx: *mut std::os::raw::c_void) -> petsc_raw::PetscErrorCode
        {
            let trampoline_data: Pin<&mut SNESFunctionTrampolineData> = std::mem::transmute(ctx);

            // We don't want to drop anything, we are just using this to turn pointers 
            // of the underlining types (i.e. *mut petsc_raw::_p_SNES) into references.
            let snes = ManuallyDrop::new(SNES::new(trampoline_data.world, snes_p));
            let x = ManuallyDrop::new(Vector { world: trampoline_data.world, vec_p: x_p });
            let mut f = ManuallyDrop::new(Vector { world: trampoline_data.world, vec_p: f_p });
            
            (trampoline_data.get_unchecked_mut().user_f)(&snes, &x, &mut f)
                .map_or_else(|err| err.kind as i32, |_| 0)
        }

        // Should we consume `input_vec`
        let ierr = unsafe { petsc_raw::SNESSetFunction(
            self.snes_p, input_vec_p, Some(snes_function_trampoline), 
            std::mem::transmute(trampoline_data.as_ref())) };
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
    ///     * `ap_mat` - the matrix to be used in constructing the (approximate) Jacobian as well as
    ///     the preconditioner.
    ///
    /// # Note
    ///
    /// You are expected to call [`Mat::assembly_begin()`] and [`Mat::assembly_end()`] at the end of
    /// `user_f`. Or you can [`Mat::assemble_with()`].
    ///
    /// # Example
    ///
    /// See example: snes ex2.rs for full code (at `examples/snes/src/ex2.rs`).
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
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
    ///     let d = f64::powi(n as f64 - 1.0, 2);
    ///
    ///     ap_mat.assemble_with((0..n).map(|i| if i == 0 || i == n-1{ vec![(i,i,1.0)] }
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
        F: FnMut(&SNES, &Vector, &mut Mat) -> Result<()> + 'tl,
    {
        // TODO: make ap_mat an `Rc<RefCell<Mat>>`

        let anchor = Box::new(user_f);

        let aj_mat_p = ap_mat.mat_p;
        let trampoline_data = unsafe {
            Pin::new_unchecked(Box::new(SNESJacobianSingleTrampolineData { 
                world: self.world, _ap_mat: ap_mat, user_f: anchor }))};
        let _ = self.jacobian_double_trampoline_data.take();
        let _ = self.jacobian_single_trampoline_data.take();

        unsafe extern "C" fn snes_jacobian_single_trampoline(snes_p: *mut petsc_raw::_p_SNES, vec_p: *mut petsc_raw::_p_Vec,
            mat1_p: *mut petsc_raw::_p_Mat, mat2_p: *mut petsc_raw::_p_Mat, ctx: *mut ::std::os::raw::c_void) -> petsc_raw::PetscErrorCode
        {
            assert_eq!(mat1_p, mat2_p);
            let trampoline_data: Pin<&mut SNESJacobianSingleTrampolineData> = std::mem::transmute(ctx);

            // We don't want to drop anything, we are just using this to turn pointers 
            // of the underlining types (i.e. *mut petsc_raw::_p_SNES) into references.
            let snes = ManuallyDrop::new(SNES::new(trampoline_data.world, snes_p));
            let x = ManuallyDrop::new(Vector { world: trampoline_data.world, vec_p: vec_p });
            let mut a_mat = ManuallyDrop::new(Mat { world: trampoline_data.world, mat_p: mat1_p });
            
            (trampoline_data.get_unchecked_mut().user_f)(&snes, &x, &mut a_mat)
                .map_or_else(|err| err.kind as i32, |_| 0)
        }

        // Should we consume `input_vec`
        let ierr = unsafe { petsc_raw::SNESSetJacobian(
            self.snes_p, aj_mat_p, aj_mat_p, Some(snes_jacobian_single_trampoline), 
            std::mem::transmute(trampoline_data.as_ref())) };
        Petsc::check_error(self.world, ierr)?;
        
        self.jacobian_single_trampoline_data = Some(trampoline_data);

        Ok(())
    }

    /// Sets the function to compute Jacobian as well as the location to store the matrix.
    ///
    /// Allows you to set a function to define what the Jacobian matrix is and when the preconditioner
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
    ///     * `a_mat` - the matrix that defines the (approximate) Jacobian.
    ///     * `p_mat` - the matrix to be used in constructing the preconditioner.
    ///
    /// # Note
    ///
    /// You are expected to call [`Mat::assembly_begin()`] and [`Mat::assembly_end()`] at the end of
    /// `user_f` on both `a_mat` and `p_mat`. Or you can [`Mat::assemble_with()`].
    ///
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
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
    /// let mut P = J.duplicate(MatDuplicateOption::MAT_SHARE_NONZERO_PATTERN)?;
    ///
    /// let mut snes = petsc.snes_create()?;
    ///
    /// snes.set_jacobian(J, P,|_snes, x: &Vector, a_mat: &mut Mat, p_mat: &mut Mat| {
    ///     let x_view = x.view()?;
    ///
    ///     let d = f64::powi(n as f64 - 1.0, 2);
    ///
    ///     a_mat.assemble_with((0..n).map(|i| if i == 0 || i == n-1{ vec![(i,i,1.0)] }
    ///                                         else { vec![(i,i-1,d), (i,i,-2.0*d+2.0*x_view[i as usize]), (i,i+1,d)] })
    ///             .flatten(), 
    ///         InsertMode::INSERT_VALUES, MatAssemblyType::MAT_FINAL_ASSEMBLY)?;
    ///
    ///     // Just set them to be the same matrix for the sake of this example
    ///     p_mat.assemble_with((0..n).map(|i| if i == 0 || i == n-1{ vec![(i,i,1.0)] }
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
        F: FnMut(&SNES, &Vector, &mut Mat, &mut Mat) -> Result<()> + 'tl
    {
        // TODO: make a/p_mat an `Rc<RefCell<Mat>>`
        let anchor = Box::new(user_f);

        let a_mat_p = a_mat.mat_p;
        let p_mat_p = p_mat.mat_p;
        let trampoline_data = unsafe {
            Pin::new_unchecked(Box::new(SNESJacobianDoubleTrampolineData { 
                world: self.world, _a_mat: a_mat, _p_mat: p_mat, user_f: anchor }))};
        let _ = self.jacobian_double_trampoline_data.take();
        let _ = self.jacobian_single_trampoline_data.take();

        unsafe extern "C" fn snes_jacobian_double_trampoline(snes_p: *mut petsc_raw::_p_SNES, vec_p: *mut petsc_raw::_p_Vec,
            mat1_p: *mut petsc_raw::_p_Mat, mat2_p: *mut petsc_raw::_p_Mat, ctx: *mut ::std::os::raw::c_void) -> petsc_raw::PetscErrorCode
        {
            assert_ne!(mat1_p, mat2_p);
            let trampoline_data: Pin<&mut SNESJacobianDoubleTrampolineData> = std::mem::transmute(ctx);

            // We don't want to drop anything, we are just using this to turn pointers 
            // of the underlining types (i.e. *mut petsc_raw::_p_SNES) into references.
            let snes = ManuallyDrop::new(SNES::new(trampoline_data.world, snes_p));
            let x = ManuallyDrop::new(Vector { world: trampoline_data.world, vec_p: vec_p });
            let mut a_mat = ManuallyDrop::new(Mat { world: trampoline_data.world, mat_p: mat1_p });
            let mut p_mat = ManuallyDrop::new(Mat { world: trampoline_data.world, mat_p: mat2_p });
            
            (trampoline_data.get_unchecked_mut().user_f)(&snes, &x, &mut a_mat, &mut p_mat)
                .map_or_else(|err| err.kind as i32, |_| 0)
        }

        // Should we consume `input_vec`
        let ierr = unsafe { petsc_raw::SNESSetJacobian(
            self.snes_p, a_mat_p, p_mat_p, Some(snes_jacobian_double_trampoline), 
            std::mem::transmute(trampoline_data.as_ref())) };
        Petsc::check_error(self.world, ierr)?;
        
        self.jacobian_double_trampoline_data = Some(trampoline_data);

        Ok(())
    }

    /// Solves linear system.
    // TODO: b
    pub fn solve(&self, b: Option<&Vector>, x: &mut Vector) -> Result<()>
    {
        let ierr = unsafe { petsc_raw::SNESSolve(self.snes_p, b.map_or(std::ptr::null_mut(), |v| v.vec_p), x.vec_p) };
        Petsc::check_error(self.world, ierr)
    }
}

// macro impls
impl<'a> SNES<'a, '_> {
    // TODO: 
    wrap_simple_petsc_member_funcs! {
        SNESSetFromOptions, set_from_options, snes_p, #[doc = "Sets various SNES and KSP parameters from user options."];
        SNESSetUp, set_up, snes_p, #[doc = "Sets up the internal data structures for the later use of a nonlinear solver. This will be automatically called with [`SNES::solve()`]."];
    }

    wrap_simple_petsc_member_funcs! {
        SNESGetIterationNumber, get_iteration_number, snes_p, i32, #[doc = "Gets the number of nonlinear iterations completed at this time. (<https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/SNES/SNESGetIterationNumber.html>)"];
    }

    wrap_simple_petsc_member_funcs! {
        SNESGetTolerances, get_tolerances, snes_p, output f64, atol, output f64, rtol, output f64, stol, output i32, maxit, output i32, maxf, #[doc = "Gets various parameters used in convergence tests."]; 
        SNESGetConvergedReason, get_converged_reason, snes_p, output SNESConvergedReason, conv_reas, #[doc = "Gets the reason the SNES iteration was stopped."];
    }
}

// TODO: Because we have two different lifetime params, these macros dont work
impl_petsc_object_funcs!{ SNES, snes_p, '_ }

impl_petsc_view_func!{ SNES, snes_p, SNESView, '_ }
