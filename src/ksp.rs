//! The scalable linear equations solvers (KSP) component provides an easy-to-use interface to the 
//! combination of a Krylov subspace iterative method and a preconditioner (in the [KSP](crate::ksp) and [PC](crate::pc)
//! components, respectively) or a sequential direct solver. 
//! 
//! KSP users can set various Krylov subspace options at runtime via the options database 
//! (e.g., -ksp_type cg ). KSP users can also set KSP options directly in application by directly calling
//! the KSP routines listed below (e.g., [`KSP::set_type()`]). KSP components can be used directly to
//! create and destroy solvers; this is not needed for users but is intended for library developers.
//!
//! PETSc C API docs: <https://petsc.org/release/docs/manualpages/KSP/index.html>

use std::ffi::CString;
use std::mem::MaybeUninit;
use std::rc::Rc;
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
    pc::{PC, },
    dm::{DM, },
};
use mpi::topology::UserCommunicator;
use mpi::traits::*;

/// [`KSP`] Type
pub use crate::petsc_raw::KSPTypeEnum as KSPType;

/// Abstract PETSc object that manages all Krylov methods. This is the object that manages the linear
/// solves in PETSc (even those such as direct solvers that do no use Krylov accelerators).
pub struct KSP<'a, 'tl, 'bl> {
    world: &'a UserCommunicator,
    pub(crate) ksp_p: *mut petsc_raw::_p_KSP, // I could use KSP which is the same thing, but i think using a pointer is more clear

    // As far as Petsc is concerned we own a reference to the PC as it is reference counted under the hood.
    // But it should be fine to keep it here as an owned reference because we can control access and the 
    // default `KSPDestroy` accounts for references.
    pub(crate) pc: Option<PC<'a, 'tl, 'bl>>,

    pub(crate) dm: Option<DM<'a, 'tl>>,
}

impl<'a, 'tl, 'bl> KSP<'a, 'tl, 'bl> {
    /// Same as `KSP { ... }` but sets all optional params to `None`
    pub(crate) fn new(world: &'a UserCommunicator, ksp_p: *mut petsc_raw::_p_KSP) -> Self {
        KSP { world, ksp_p, pc: None, dm: None, }
    }

    /// Same as [`Petsc::ksp_create()`](crate::Petsc::ksp_create).
    pub fn create(world: &'a UserCommunicator) -> Result<Self> {
        let mut ksp_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::KSPCreate(world.as_raw(), ksp_p.as_mut_ptr()) };
        unsafe { chkerrq!(world, ierr) }?;

        Ok(KSP::new(world, unsafe { ksp_p.assume_init() }))
    }

    /// Builds [`KSP`] for a particular solver. (given as `&str`).
    pub fn set_type_str(&mut self, ksp_type: &str) -> Result<()> {
        let cstring = CString::new(ksp_type).expect("`CString::new` failed");
        let ierr = unsafe { petsc_raw::KSPSetType(self.ksp_p, cstring.as_ptr()) };
        unsafe { chkerrq!(self.world, ierr) }
    }

    /// Builds [`KSP`] for a particular solver.
    pub fn set_type(&mut self, ksp_type: KSPType) -> Result<()> {
        let cstring = petsc_raw::KSPTYPE_TABLE[ksp_type as usize];
        let ierr = unsafe { petsc_raw::KSPSetType(self.ksp_p, cstring.as_ptr() as *const _) };
        unsafe { chkerrq!(self.world, ierr) }
    }

    /// Sets the matrix associated with the linear system and a (possibly)
    /// different one associated with the preconditioner. Note, this is the same as
    /// `ksp.get_pc_or_create().set_operators()`.
    ///
    /// Passing a `None` for `a_mat` or `p_mat` removes the matrix that is currently used.
    pub fn set_operators(&mut self, a_mat: impl Into<Option<&'bl Mat<'a, 'tl>>>, p_mat: impl Into<Option<&'bl Mat<'a, 'tl>>>) -> Result<()> {
        self.get_pc_or_create()?.set_operators(a_mat, p_mat)?;
        // This is done in the C API `KSPSetOperators` function.
        if unsafe { &mut *self.ksp_p }.setupstage == petsc_raw::KSPSetUpStage::KSP_SETUP_NEWRHS {
            // so that next solve call will call PCSetUp() on new matrix
            unsafe { &mut *self.ksp_p }.setupstage = petsc_raw::KSPSetUpStage::KSP_SETUP_NEWMATRIX;
        }
        Ok(())
    }
    
    /// Returns a [`Option`] of a reference to the [`PC`] context set.
    ///
    /// If you want PETSc to set the [`PC`] you must call [`KSP::set_pc()`]
    /// or [`KSP::get_pc_or_create()`].
    ///
    /// Note, this does not return a [`Result`](crate::Result) because it can never
    /// fail, instead it will return `None`.
    pub fn try_get_pc<'b>(&'b self) -> Option<&'b PC<'a, 'tl, 'bl>> {
        self.pc.as_ref()
    }

    /// Returns a mutable reference to the [`PC`] context set with [`KSP::set_pc()`],
    /// or creates a default one if no [`PC`] has been set.
    pub fn get_pc_or_create<'b>(&'b mut self) -> Result<&'b mut PC<'a, 'tl, 'bl>>
    {
        if self.pc.is_some() {
            Ok(self.pc.as_mut().unwrap())
        } else {
            let mut pc_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::KSPGetPC(self.ksp_p, pc_p.as_mut_ptr()) };
            unsafe { chkerrq!(self.world, ierr) }?;

            self.pc = Some(PC::new(self.world, unsafe { pc_p.assume_init() }));
            unsafe { self.pc.as_mut().unwrap().reference()?; }

            Ok(self.pc.as_mut().unwrap())
        }
    }

    /// Returns an [`Option`] to a reference to the [DM](DM).
    ///
    /// If you want PETSc to set the [`DM`] you must call
    /// [`KSP::get_dm_or_create()`], otherwise you must call [`KSP::set_dm()`]
    /// for this to return a `Some`.
    ///
    /// Note, this does not return a [`Result`](crate::Result) because it can never
    /// fail, instead it will return `None`.
    pub fn try_get_dm<'b>(&'b self) -> Option<&'b DM<'a, 'tl>> {
        self.dm.as_ref()
    }

    /// Returns a mutable reference to the [DM](DM) that may be used by some [preconditioners](crate::pc),
    /// or creates a default one if no [`DM`] has been set.
    pub fn get_dm_or_create<'b>(&'b mut self) -> Result<&'b mut DM<'a, 'tl>>
    {
        if self.dm.is_some() {
            Ok(self.dm.as_mut().unwrap())
        } else {
            let mut dm_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::KSPGetDM(self.ksp_p, dm_p.as_mut_ptr()) };
            unsafe { chkerrq!(self.world, ierr) }?;

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
        unsafe { chkerrq!(self.world, ierr) }
    }

    /// Solves linear system.
    pub fn solve<'vl, 'val: 'vl>(&mut self, b: impl Into<Option<&'vl Vector<'val>>>, x: &mut Vector) -> Result<()>
    {
        let ierr = unsafe { petsc_raw::KSPSolve(self.ksp_p, b.into().map_or(std::ptr::null_mut(), |v| v.vec_p), x.vec_p) };
        unsafe { chkerrq!(self.world, ierr) }
    }

    /// Sets the routine to compute the linear operators
    ///
    /// The user provided func() will be called automatically at the very next call to [`KSP::solve()`].
    /// It will not be called at future [`KSP::solve()`] calls unless either KSPSetComputeOperators()
    /// or KSPSetOperators() is called before that [`KSP::solve()`] is called.
    ///
    /// To reuse the same preconditioner for the next [`KSP::solve()`] and not compute a new one based
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
    /// [`let dm = ksp.try_get_dm().unwrap();`](KSP::try_get_dm()).
    pub fn set_compute_operators<F>(&mut self, user_f: F) -> Result<()>
    where
        F: FnMut(&KSP<'a, '_, '_>, &mut Mat<'a, '_>, &mut Mat<'a, '_>) -> Result<()> + 'tl
    {
        let ierr = unsafe { petsc_raw::KSPSetComputeOperators(
            self.ksp_p, None, std::ptr::null_mut()) };
        unsafe { chkerrq!(self.world, ierr) }?;
        
        self.get_dm_or_create()?.ksp_set_compute_operators(user_f)
    }

    /// Sets the routine to compute the right hand side of the linear system
    ///
    /// The routine you provide will be called EACH time you call [`KSP::solve()`] to prepare the
    /// new right hand side for that solve
    ///
    /// # Parameters
    ///
    /// * `user_f` - A closure used to convey the routine to compute the the right hand side of the linear system
    ///     * `ksp` - the ksp context
    ///     * `b` *(output)* - right hand side of linear system
    ///
    /// # Note
    ///
    /// You can access the [`DM`] owned by the `ksp` in the `user_f` by using
    /// [`let dm = ksp.try_get_dm().unwrap();`](KSP::try_get_dm()).
    pub fn set_compute_rhs<F>(&mut self, user_f: F) -> Result<()>
    where
        F: FnMut(&KSP<'a, '_, '_>, &mut Vector<'a>) -> Result<()> + 'tl
    {
        let ierr = unsafe { petsc_raw::KSPSetComputeRHS(
            self.ksp_p, None, std::ptr::null_mut()) };
        unsafe { chkerrq!(self.world, ierr) }?;
        
        self.get_dm_or_create()?.ksp_set_compute_rhs(user_f)
    }

    /// Gets the right-hand-side vector for the linear system to be solved.
    /// TODO: Is it safe to return an Rc
    pub fn get_rhs(&self) -> Result<Rc<Vector<'a>>> {
        let mut vec_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::KSPGetRhs(self.ksp_p, vec_p.as_mut_ptr()) };
        unsafe { chkerrq!(self.world, ierr) }?;

        let mut vec = Vector { world: self.world, vec_p: unsafe { vec_p.assume_init() } };
        unsafe { vec.reference()?; }
        let rhs = Rc::new(vec);

        Ok(rhs)
    }

    /// Gets the matrix associated with the linear system and possibly a different
    /// one associated with the preconditioner.
    ///
    /// See also [`PC::get_operators_or_create()`].
    ///
    /// If the operators have NOT been set with [`KSP`](KSP::set_operators())/[`PC::set_operators()`]
    /// then the operators are created in the PC and returned to the user. In this case, two DIFFERENT
    /// operators will be returned.
    pub fn get_operators_or_create<'rl>(&'rl mut self) -> Result<(&'rl Mat<'a, 'tl>, &'rl Mat<'a, 'tl>)> {
        self.get_pc_or_create()?.get_operators_or_create()
    }

    /// Gets the matrix associated with the linear system and possibly a different
    /// one associated with the preconditioner.
    ///
    /// See also [`PC::get_operators_or_create()`].
    ///
    /// If the operators have NOT been set with [`KSP`](crate::ksp::KSP::set_operators())/[`PC::set_operators()`](crate::pc::PC::set_operators())
    /// then this will return `None` for those operators. Also, if the PC is not set it will return `None`s,
    /// this is because it uses [`KSP::try_get_pc()`].
    ///
    /// Note, if you used [`KSP::set_compute_operators()`] to set the operators, you must use
    /// [`KSP::get_operators_or_create()`] to create the operators from the method.
    pub fn try_get_operators<'rl>(&'rl self) -> Result<(Option<&'rl Mat<'a, 'tl>>, Option<&'rl Mat<'a, 'tl>>)> {
        if let Some(pc) = self.try_get_pc() {
            pc.try_get_operators()
        } else {
            Ok((None, None))
        }
    }

    /// Determines whether a PETSc [`KSP`] is of a particular type.
    pub fn type_compare(&self, type_kind: KSPType) -> Result<bool> {
        self.type_compare_str(&type_kind.to_string())
    }
}

// macro impls
impl<'a, 'tl, 'bl> KSP<'a, 'tl, 'bl> {
    wrap_simple_petsc_member_funcs! {
        KSPSetFromOptions, pub set_from_options, takes mut, #[doc = "Sets KSP options from the options database. This routine must be called before KSPSetUp() if the user is to be allowed to set the Krylov type."];
        KSPSetUp, pub set_up, takes mut, #[doc = "Sets up the internal data structures for the later use of an iterative solver. . This will be automatically called with [`KSP::solve()`]."];
        KSPGetIterationNumber, pub get_iteration_number, output PetscInt, iter_num, #[doc = "Gets the current iteration number; if the [`KSP::solve()`] is complete, returns the number of iterations used."];
        KSPSetDM, pub set_dm, input DM<'a, 'tl>, dm .as_raw consume .dm, takes mut, #[doc = "Sets the [DM](DM) that may be used by some [preconditioners](crate::pc).\n\n\
            If this is used then the KSP will attempt to use the DM to create the matrix and use the routine set with [`DMKSPSetComputeOperators()`](#) or [`KSP::set_compute_operators()`]. Use\
            [`KSP::set_dm_active(false)`](KSP::set_dm_active()) to instead use the matrix you've provided with [`KSP::set_operators()`]."];
        KSPSetPC, pub set_pc, input PC<'a, 'tl, 'bl>, pc .as_raw consume .pc, takes mut, #[doc = "Sets the [preconditioner](crate::pc)([`PC`]) to be used to calculate the application of the preconditioner on a vector.\n\n\
            If you change the PC by calling set again, then the original will be dropped."];
        KSPSetDMActive, pub set_dm_active, input bool, flg, takes mut, #[doc = "Indicates that the [`DM`] should be used to generate the linear system matrix and right hand side."]; 
    }
}

impl_petsc_object_traits! { KSP, ksp_p, petsc_raw::_p_KSP, KSPView, KSPDestroy, '_, '_; }
