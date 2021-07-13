//! The preconditioner used by the [KSP](crate::ksp).
//!
//! KSP users can set various preconditioning options at runtime via the options database 
//! (e.g., -pc_type jacobi ). KSP users can also set PC options directly in application codes by 
//! first extracting the PC context from the KSP context via [`KSP::get_pc()`](crate::ksp::KSP::get_pc()) and then directly
//! calling the PC routines listed below (e.g., [`PC::set_type()`]). PC components can be used directly
//! to create and destroy solvers; this is not needed for users but is for library developers.
//!
//! PETSc C API docs: <https://petsc.org/release/docs/manualpages/PC/index.html>

use std::mem::{MaybeUninit, ManuallyDrop};
use std::ffi::{CString, };
use std::rc::Rc;
use std::pin::Pin;
use crate::{
    Petsc,
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

pub use crate::petsc_raw::PCTypeEnum as PCType;

/// Abstract PETSc object that manages all preconditioners including direct solvers such as PCLU
pub struct PC<'a, 'tl> {
    pub(crate) world: &'a UserCommunicator,
    pub(crate) pc_p: *mut petsc_raw::_p_PC, // I could use petsc_raw::PC which is the same thing, but i think using a pointer is more clear

    // We take an `Rc` because we don't want ownership of the Mat. Under the hood, this is how the 
    // PetscSetOperators function works, it increments the reference count. The problem with this
    // solution right now is that we lose mutable access. It might be worth making it a Rc<RefCell<Mat>>.
    // This might also allow us to have a get_operators function (which would also returns a Rc<RefCell<Mat>>).
    // Regardless, returning mutable access would be hard, especially when the rust side can't guarantee how the 
    // C api accesses the operators behind the scenes.
    ref_amat: Option<Rc<Mat<'a>>>,
    ref_pmat: Option<Rc<Mat<'a>>>,

    shell_set_apply_trampoline_data: Option<Pin<Box<PCShellSetApplyTrampolineData<'a, 'tl>>>>,
}

struct PCShellSetApplyTrampolineData<'a, 'tl> {
    world: &'a UserCommunicator,
    user_f: Box<dyn FnMut(&PC<'a, 'tl>, &Vector<'a>, &mut Vector<'a>) -> Result<()> + 'tl>,
}

impl<'a> Drop for PC<'a, '_> {
    // Note, this should only be called if the PC context was created with `PCCreate`.
    fn drop(&mut self) {
        let ierr = unsafe { petsc_raw::PCDestroy(&mut self.pc_p as *mut _) };
        let _ = Petsc::check_error(self.world, ierr); // TODO: should I unwrap or what idk?
    }
}

impl<'a, 'tl> PC<'a, 'tl> {
    /// Same as `PC { ... }` but sets all optional params to `None`
    pub(crate) fn new(world: &'a UserCommunicator, pc_p: *mut petsc_raw::_p_PC) -> Self {
        PC { world, pc_p, ref_amat: None, ref_pmat: None,
            shell_set_apply_trampoline_data: None }
    }

    /// Creates a preconditioner context.
    ///
    /// You will most likely create a preconditioner context from a solver type such as
    /// from a Krylov solver, [`KSP`](crate::ksp::KSP), using the [`KSP::get_pc()`](crate::ksp::KSP::get_pc()) method.
    ///
    /// [`KSP::get_pc`]: KSP::get_pc
    pub fn create(world: &'a UserCommunicator) -> Result<Self> {
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
    pub fn set_operators(&mut self, a_mat: impl Into<Option<Rc<Mat<'a>>>>, p_mat: impl Into<Option<Rc<Mat<'a>>>>) -> Result<()> {
        let a_mat = a_mat.into();
        let p_mat = p_mat.into();
        // TODO: should we make a_mat an `Rc<RefCell<Mat>>`, `Rc<Mat>`, or just a `Mat`
        // if this consumes a_mat and p_mat, make `set_operators_single_mat` so that you can set
        // them to be the same.

        // Should this function consume the mats? Right now you have to turn the mats into `Rc`s which
        // means you loose mutable access, even if you remove them.

        let ierr = unsafe { petsc_raw::PCSetOperators(self.pc_p,
            a_mat.as_ref().map_or(std::ptr::null_mut(), |m| m.mat_p), 
            p_mat.as_ref().map_or(std::ptr::null_mut(), |m| m.mat_p)) };
        Petsc::check_error(self.world, ierr)?;

        // drop everything as it is getting replaced. (note under the hood MatDestroy is called on both of
        // them each time `PCSetOperators` is called).
        let _ = self.ref_amat.take();
        let _ = self.ref_pmat.take();

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
    pub fn get_operators(&self) -> Result<(Rc<Mat<'a>>, Rc<Mat<'a>>)> {
        // TODO: maybe this should return Rc<RefCell<T>> so that the caller can edit the matrices
        // https://petsc.org/release/docs/manualpages/PC/PCGetOperators.html#PCGetOperators
        // Although that would mean set_operators should also take Rc<RefCell<T>>

        let a_mat = if let Some(ref a_mat) = self.ref_amat {
            a_mat.clone()
        }
        else
        {
            let mut a_mat_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::PCGetOperators(self.pc_p, a_mat_p.as_mut_ptr(), std::ptr::null_mut()) };
            Petsc::check_error(self.world, ierr)?;

            let mut mat = Mat { world: self.world, mat_p: unsafe { a_mat_p.assume_init() } };
            unsafe { mat.reference()?; }
            Rc::new(mat)
        };

        let p_mat = if let Some(ref p_mat) = self.ref_pmat {
            p_mat.clone()
        }
        else
        {
            let mut p_mat_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::PCGetOperators(self.pc_p, std::ptr::null_mut(), p_mat_p.as_mut_ptr()) };
            Petsc::check_error(self.world, ierr)?;

            let mut mat = Mat { world: self.world, mat_p: unsafe { p_mat_p.assume_init() } };
            unsafe { mat.reference()?; }
            Rc::new(Mat { world: self.world, mat_p: unsafe { p_mat_p.assume_init() } })
        };

        Ok((a_mat, p_mat))
    }

    // TODO: make a branch to test this out
    // /// Sets the matrix associated with the linear system and a (possibly)
    // /// different one associated with the preconditioner.
    // ///
    // /// Note, this method will borrow `a_mat` and `p_mat` as mutable until they are dropped.
    // /// Passing a `None` for `a_mat` or `p_mat` removes the matrix that is currently used.
    // /// This will drop the mutable borrows taken from the `RefCells` giving mutable access
    // /// back to the caller.
    // pub fn set_operators(&mut self, a_mat: Option<Rc<RefCell<Mat<'a>>>>, p_mat: Option<Rc<RefCell<Mat<'a>>>>) -> Result<()>
    // {
    //     // drop everything as it is getting replaced. (note under the hood MatDestroy is called on both of
    //     // them each time `PCSetOperators` is called).
    //     // let _ = self.ref_amat.take();
    //     // let _ = self.ref_pmat.take();
    //
    //     let ierr = unsafe { petsc_raw::PCSetOperators(self.pc_p,
    //         a_mat.as_ref().map_or(std::ptr::null_mut(), |m| m.borrow().mat_p), 
    //         p_mat.as_ref().map_or(std::ptr::null_mut(), |m| m.borrow().mat_p)) };
    //     Petsc::check_error(self.world, ierr)?;
    //
    //     let _ = self.a_mat.take();
    //     let _ = self.p_mat.take();
    //
    //     self.a_mat = a_mat;
    //     self.p_mat = p_mat;
    //     // self.ref_amat = self.a_mat.map(|m| m.borrow_mut());
    //     // self.ref_pmat = self.p_mat.map(|m| m.borrow_mut());
    //
    //     Ok(())
    // }

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
        F: FnMut(&PC<'a, 'tl>, &Vector<'a>, &mut Vector<'a>) -> Result<()> + 'tl
    {
        // TODO: look at how rsmpi did the trampoline stuff:
        // https://github.com/rsmpi/rsmpi/blob/master/src/collective.rs#L1684
        // They used libffi, that could be a safer way to do it.

        let closure_anchor = Box::new(user_f);

        let trampoline_data = Box::pin(PCShellSetApplyTrampolineData { 
            world: self.world, user_f: closure_anchor });

        // drop old trampoline_data
        let _ = self.shell_set_apply_trampoline_data.take();

        unsafe extern "C" fn pc_shell_set_apply_trampoline(pc_p: *mut petsc_raw::_p_PC, xin_p: *mut petsc_raw::_p_Vec,
            xout_p: *mut petsc_raw::_p_Vec) -> petsc_raw::PetscErrorCode
        {
            let mut ctx = MaybeUninit::uninit();
            let ierr = petsc_raw::PCShellGetContext(pc_p, ctx.as_mut_ptr());
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
            
            (trampoline_data.get_unchecked_mut().user_f)(&pc, &xin, &mut xout)
                .map_or_else(|err| err.kind as i32, |_| 0)
        }

        let ierr = unsafe { petsc_raw::PCShellSetApply(
            self.pc_p, Some(pc_shell_set_apply_trampoline)) };
        Petsc::check_error(self.world, ierr)?;
        let ierr = unsafe { petsc_raw::PCShellSetContext(self.pc_p,
            std::mem::transmute(trampoline_data.as_ref())) }; // this will also erase the lifetimes
        Petsc::check_error(self.world, ierr)?;
        
        self.shell_set_apply_trampoline_data = Some(trampoline_data);

        Ok(())
    }

    /// Sets the exact elements for field 
    ///
    /// # Parameters
    ///
    /// * `splitname` - name of this split, if `None` the number of the split is used.
    /// * `is` - the index set that defines the vector elements in this field 
    pub fn field_split_set_is<T: ToString>(&mut self, splitname: Option<T>, is: IS) -> Result<()> {
        let splitname_cs = splitname.map(|to_str|
            CString::new(to_str.to_string()).expect("`CString::new` failed"));

        let ierr = unsafe { petsc_raw::PCFieldSplitSetIS(
            self.pc_p, splitname_cs.map(|cs| cs.as_ptr()).unwrap_or(std::ptr::null()), is.is_p) };
        Petsc::check_error(self.world, ierr)
    }
}

// Macro impls
impl<'a> PC<'a, '_> {
    wrap_simple_petsc_member_funcs! {
        PCSetFromOptions, pub set_from_options, takes mut, #[doc = "Sets PC options from the options database. This routine must be called before PCSetUp() if the user is to be allowed to set the preconditioner method."];
        PCSetUp, pub set_up, takes mut, #[doc = "Prepares for the use of a preconditioner."];
    }
}

impl_petsc_object_traits! { PC, pc_p, petsc_raw::_p_PC, '_ }

impl_petsc_view_func!{ PC, PCView, '_ }
