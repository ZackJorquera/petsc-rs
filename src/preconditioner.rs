//! The preconditioner used by the [KSP](crate::ksp).
//!
//! KSP users can set various preconditioning options at runtime via the options database 
//! (e.g., -pc_type jacobi ). KSP users can also set PC options directly in application codes by 
//! first extracting the PC context from the KSP context via [`KSP::get_pc_or_create()`](crate::ksp::KSP::get_pc_or_create()) and then directly
//! calling the PC routines listed below (e.g., [`PC::set_type()`]). PC components can be used directly
//! to create and destroy solvers; this is not needed for users but is for library developers.
//!
//! PETSc C API docs: <https://petsc.org/release/docs/manualpages/PC/index.html>

use std::mem::{MaybeUninit, ManuallyDrop};
use std::ffi::{CString, };
use std::pin::Pin;
use crate::{
    petsc_raw,
    Result,
    PetscAsRaw,
    PetscObject,
    PetscObjectPrivate,
    vector::{Vector, },
    mat::{Mat, },
    indexset::{IS, },
};
use mpi::topology::UserCommunicator;
use mpi::traits::*;

/// [`PC`] Type
pub use crate::petsc_raw::PCTypeEnum as PCType;

/// Abstract PETSc object that manages all preconditioners including direct solvers such as PCLU
pub struct PC<'a, 'tl, 'bl> {
    pub(crate) world: &'a UserCommunicator,
    pub(crate) pc_p: *mut petsc_raw::_p_PC, // I could use petsc_raw::PC which is the same thing, but i think using a pointer is more clear

    // Note, to prevent self references, the ref version is `None`
    // if the current mat value is stored in the owned version
    ref_amat: Option<&'bl Mat<'a, 'tl>>,
    ref_pmat: Option<&'bl Mat<'a, 'tl>>,
    owned_amat: Option<Mat<'a, 'tl>>,
    owned_pmat: Option<Mat<'a, 'tl>>,

    shell_set_apply_trampoline_data: Option<Pin<Box<PCShellSetApplyTrampolineData<'a, 'tl>>>>,
}

struct PCShellSetApplyTrampolineData<'a, 'tl> {
    world: &'a UserCommunicator,
    user_f: Box<dyn FnMut(&PC<'a, 'tl, '_>, &Vector<'a>, &mut Vector<'a>) -> Result<()> + 'tl>,
}

impl<'a, 'tl, 'bl> PC<'a, 'tl, 'bl> {
    /// Same as `PC { ... }` but sets all optional params to `None`
    pub(crate) fn new(world: &'a UserCommunicator, pc_p: *mut petsc_raw::_p_PC) -> Self {
        PC { world, pc_p, ref_amat: None, ref_pmat: None, owned_amat: None, owned_pmat: None,
            shell_set_apply_trampoline_data: None }
    }

    /// Creates a preconditioner context.
    ///
    /// You will most likely create a preconditioner context from a solver type such as
    /// from a Krylov solver, [`KSP`](crate::ksp::KSP), using the
    /// [`KSP::get_pc_or_create()`](crate::ksp::KSP::get_pc_or_create()) method.
    pub fn create(world: &'a UserCommunicator) -> Result<Self> {
        let mut pc_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::PCCreate(world.as_raw(), pc_p.as_mut_ptr()) };
        unsafe { chkerrq!(world, ierr) }?;

        Ok(PC::new(world, unsafe { pc_p.assume_init() }))
    }

    /// Builds [`PC`] for a particular preconditioner type (given as `&str`).
    pub fn set_type_str(&mut self, pc_type: &str) -> Result<()> {
        let cstring = CString::new(pc_type).expect("`CString::new` failed");
        let ierr = unsafe { petsc_raw::PCSetType(self.pc_p, cstring.as_ptr()) };
        unsafe { chkerrq!(self.world, ierr) }
    }

    /// Builds [`PC`] for a particular preconditioner type
    pub fn set_type(&mut self, pc_type: PCType) -> Result<()>
    {
        let cstring = petsc_raw::PCTYPE_TABLE[pc_type as usize];
        let ierr = unsafe { petsc_raw::PCSetType(self.pc_p, cstring.as_ptr() as *const _) };
        unsafe { chkerrq!(self.world, ierr) }
    }
    
    /// Sets the matrix associated with the linear system and a (possibly)
    /// different one associated with the preconditioner.
    ///
    /// Passing a `None` for `a_mat` or `p_mat` removes the matrix that is currently used.
    pub fn set_operators(&mut self, a_mat: impl Into<Option<&'bl Mat<'a, 'tl>>>, p_mat: impl Into<Option<&'bl Mat<'a, 'tl>>>) -> Result<()> {
        let a_mat = a_mat.into();
        let p_mat = p_mat.into();

        let ierr = unsafe { petsc_raw::PCSetOperators(self.pc_p,
            a_mat.as_ref().map_or(std::ptr::null_mut(), |m| m.mat_p), 
            p_mat.as_ref().map_or(std::ptr::null_mut(), |m| m.mat_p)) };
        unsafe { chkerrq!(self.world, ierr) }?;

        // drop everything as it is getting replaced. (note under the hood MatDestroy is called on both of
        // them each time `PCSetOperators` is called).
        let _ = self.ref_amat.take();
        let _ = self.ref_pmat.take();
        // We aren't using the owned mats anymore so drop them.
        let _ = self.owned_amat.take();
        let _ = self.owned_pmat.take();

        self.ref_amat = a_mat;
        self.ref_pmat = p_mat;

        Ok(())
    }
    
    /// Gets the matrix associated with the linear system and possibly a different
    /// one associated with the preconditioner.
    ///
    /// If the operators have NOT been set with [`KSP`](crate::ksp::KSP::set_operators())/[`PC::set_operators()`](crate::pc::PC::set_operators())
    /// then the operators are created in the PC and returned to the user. In this case, two DIFFERENT
    /// operators will be returned.
    pub fn get_operators_or_create<'rl>(&'rl mut self) -> Result<(&'rl Mat<'a, 'tl>, &'rl Mat<'a, 'tl>)> {
        let a_mat = if let Some(a_mat) = self.ref_amat {
            a_mat
        }
        else
        {
            if self.owned_amat.is_none() {
                let mut a_mat_p = MaybeUninit::zeroed();
                let ierr = unsafe { petsc_raw::PCGetOperators(self.pc_p, a_mat_p.as_mut_ptr(), std::ptr::null_mut()) };
                unsafe { chkerrq!(self.world, ierr) }?;

                let mut mat = Mat::new(self.world, unsafe { a_mat_p.assume_init() });
                // We only call this if amat has not been set which means that PETSc will create a new mat
                // so it is ok to increment the reference of mat and take ownership.
                unsafe { mat.reference()?; }
                self.owned_amat = Some(mat);
            }
            self.owned_amat.as_ref().unwrap()
        };

        let p_mat = if let Some(p_mat) = self.ref_pmat {
            p_mat
        }
        else
        {
            if self.owned_pmat.is_none() {
                let mut p_mat_p = MaybeUninit::zeroed();
                let ierr = unsafe { petsc_raw::PCGetOperators(self.pc_p, std::ptr::null_mut(), p_mat_p.as_mut_ptr()) };
                unsafe { chkerrq!(self.world, ierr) }?;

                let mut mat = Mat::new(self.world, unsafe { p_mat_p.assume_init() });
                // We only call this if pmat has not been set which means that PETSc will create a new mat
                // so it is ok to increment the reference of mat and take ownership.
                unsafe { mat.reference()?; }
                self.owned_pmat = Some(mat);
            }
            self.owned_pmat.as_ref().unwrap()
        };

        Ok((a_mat, p_mat))
    }

    /// Gets the matrix associated with the linear system and possibly a different
    /// one associated with the preconditioner.
    ///
    /// If the operators have NOT been set with [`KSP`](crate::ksp::KSP::set_operators())/[`PC::set_operators()`](crate::pc::PC::set_operators())
    /// then this will return `None` for those operators.
    ///
    /// Note, if you used [`KSP::set_compute_operators()`](crate::ksp::KSP::set_compute_operators()) to set the operators, you must use
    /// [`PC::get_operators_or_create()`] to create the operators from the method.
    pub fn try_get_operators<'rl>(&'rl self) -> Result<(Option<&'rl Mat<'a, 'tl>>, Option<&'rl Mat<'a, 'tl>>)> {
        Ok((
            if self.ref_amat.is_some() {
                self.ref_amat
            } else { 
                self.owned_amat.as_ref()
            },
            if self.ref_pmat.is_some() {
                self.ref_pmat
            } else {
                self.owned_pmat.as_ref()
            }
        ))
    }

    /// Sets routine to use as preconditioner. 
    ///
    /// # Parameters
    ///
    /// * `user_f` - A closure used to convey the Jacobian evaluation routine.
    ///     * `pc` - the preconditioner context
    ///     * `xin` - input vector
    ///     * `xout` *(output)* - output vector
    pub fn shell_set_apply<F>(&mut self, user_f: F) -> Result<()>
    where
        F: FnMut(&PC<'a, 'tl, '_>, &Vector<'a>, &mut Vector<'a>) -> Result<()> + 'tl
    {
        // TODO: look at how rsmpi did the trampoline stuff:
        // https://github.com/rsmpi/rsmpi/blob/82e1d357/src/collective.rs#L1684
        // They used libffi, that could be a safer way to do it.

        let closure_anchor = Box::new(user_f);

        let trampoline_data = Box::pin(PCShellSetApplyTrampolineData { 
            world: self.world, user_f: closure_anchor });

        // drop old trampoline_data
        let _ = self.shell_set_apply_trampoline_data.take();

        unsafe extern "C" fn pc_shell_set_apply_trampoline(pc_p: *mut petsc_raw::_p_PC, xin_p: *mut petsc_raw::_p_Vec,
            xout_p: *mut petsc_raw::_p_Vec) -> petsc_raw::PetscErrorCode
        {
            // Note, the function signature of PCShellGetContext has changed in v3.16-dev.0,
            // The following should work for both v3.16-dev.0 and v3.15
            let mut ctx = MaybeUninit::<*mut ::std::os::raw::c_void>::uninit();
            let ierr = petsc_raw::PCShellGetContext(pc_p, ctx.as_mut_ptr() as *mut _);
            assert_eq!(ierr, 0);

            // SAFETY: We construct ctx to be a Pin<Box<KSPComputeOperatorsTrampolineData>> but pass it in as a *void
            // Box<T> is equivalent to *T (or &T) for ffi. Because the KSP owns the closure we can make sure
            // everything in it (and the closure its self) lives for at least as long as this function can be
            // called.
            // We don't construct a Box<> because we dont want to drop anything
            let trampoline_data: Pin<&mut PCShellSetApplyTrampolineData> = std::mem::transmute(ctx.assume_init());

            // We don't want to drop anything, we are just using this to turn pointers 
            // of the underlining types (i.e. *mut petsc_raw::_p_SNES) into references
            // of the rust wrapper types.
            // If `Vector` ever has optional parameters, they MUST be dropped manually.
            let pc = ManuallyDrop::new(PC::new(trampoline_data.world, pc_p));
            let xin = ManuallyDrop::new(Vector { world: trampoline_data.world, vec_p: xin_p });
            let mut xout = ManuallyDrop::new(Vector { world: trampoline_data.world, vec_p: xout_p });
            
            (trampoline_data.get_mut().user_f)(&pc, &xin, &mut xout)
                .map_or_else(|err| err.kind as i32, |_| 0)
        }

        let ierr = unsafe { petsc_raw::PCShellSetApply(
            self.pc_p, Some(pc_shell_set_apply_trampoline)) };
        unsafe { chkerrq!(self.world, ierr) }?;
        let ierr = unsafe { petsc_raw::PCShellSetContext(self.pc_p,
            std::mem::transmute(trampoline_data.as_ref())) }; // this will also erase the lifetimes
        unsafe { chkerrq!(self.world, ierr) }?;
        
        self.shell_set_apply_trampoline_data = Some(trampoline_data);

        Ok(())
    }

    /// Sets the exact elements for field 
    ///
    /// # Parameters
    ///
    /// * `splitname` - name of this split, if `None` the number of the split is used.
    /// * `is` - the index set that defines the vector elements in this field 
    pub fn field_split_set_is<'strl>(&mut self, splitname: impl Into<Option<&'strl str>>, is: IS) -> Result<()> {
        let splitname_cs = splitname.into().map(|to_str|
            CString::new(to_str.to_string()).expect("`CString::new` failed"));

        let ierr = unsafe { petsc_raw::PCFieldSplitSetIS(
            self.pc_p, splitname_cs.map(|cs| cs.as_ptr()).unwrap_or(std::ptr::null()), is.is_p) };
        unsafe { chkerrq!(self.world, ierr) }
    }

    /// Determines whether a PETSc [`PC`] is of a particular type.
    pub fn type_compare(&self, type_kind: PCType) -> Result<bool> {
        self.type_compare_str(&type_kind.to_string())
    }
}

// Macro impls
impl<'a> PC<'a, '_, '_> {
    wrap_simple_petsc_member_funcs! {
        PCSetFromOptions, pub set_from_options, takes mut, #[doc = "Sets PC options from the options database. This routine must be called before PCSetUp() if the user is to be allowed to set the preconditioner method."];
        PCSetUp, pub set_up, takes mut, #[doc = "Prepares for the use of a preconditioner."];
    }
}

impl_petsc_object_traits! { PC, pc_p, petsc_raw::_p_PC, PCView, PCDestroy, '_, '_; }
