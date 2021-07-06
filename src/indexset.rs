//! Index set (IS) objects are used to index into vectors and matrices and to setup vector scatters.
//!
//! PETSc C API docs: <https://petsc.org/release/docs/manualpages/IS/index.html>

use crate::prelude::*;

/// Abstract PETSc object that allows indexing. 
pub struct IS<'a> {
    pub(crate) world: &'a dyn Communicator,

    pub(crate) is_p: *mut petsc_raw::_p_IS,
}

impl<'a> Drop for IS<'a> {
    fn drop(&mut self) {
        let ierr = unsafe { petsc_raw::ISDestroy(&mut self.is_p as *mut _) };
        let _ = Petsc::check_error(self.world, ierr); // TODO: should I unwrap or what idk?
    }
}

impl<'a> IS<'a> {
    /// Creates an index set object. 
    pub fn create(world: &'a dyn Communicator) -> Result<Self> {
        let mut is_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::ISCreate(world.as_raw(), is_p.as_mut_ptr()) };
        Petsc::check_error(world, ierr)?;

        Ok(IS { world, is_p: unsafe { is_p.assume_init() } })
    }

    // wrap_simple_petsc_member_funcs! {
    // }
}

impl_petsc_object_funcs!{ IS, is_p }

impl_petsc_view_func!{ IS, is_p, ISView }
