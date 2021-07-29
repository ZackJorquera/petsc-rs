//! PETSc viewers export information and data from PETSc objects.
//!
//! PETSc C API docs: <https://petsc.org/release/docs/manualpages/Viewer/index.html>

use std::mem::MaybeUninit;
use crate::{
    Petsc,
    petsc_raw,
    Result,
    PetscAsRaw,
    PetscObject,
    PetscObjectPrivate
};
use mpi::topology::UserCommunicator;
use mpi::traits::*;

pub use petsc_sys::PetscViewerFormat;

/// [`Viewer`] Type
pub type ViewerType = crate::petsc_raw::ViewerTypeEnum;

/// Abstract collection of PetscViewers. It is just an expandable array of viewers.
// TODO: right now this is a very basic wrapper of the view functionality, I feel like
// we could do this more rusty, but i don't really know how.
// It might make sense to make our own viewer code. That we can do everything on the rust side
// This would work more nicely with things like display and debug, also the file system.
// for now we will just use the C api.
pub struct Viewer<'a> {
    world: &'a UserCommunicator,

    pub(crate) viewer_p: *mut petsc_raw::_p_PetscViewer,
}

impl<'a> Drop for Viewer<'a> {
    fn drop(&mut self) {
        let ierr = unsafe { petsc_raw::PetscViewerDestroy(&mut self.viewer_p as *mut _) };
        let _ = Petsc::check_error(self.world, ierr); // TODO: should I unwrap or what idk?
    }
}

impl<'a> Viewer<'a> {
    /// Creates a ASCII PetscViewer shared by all processors in a communicator.
    pub fn create_ascii_stdout(world: &'a UserCommunicator) -> Result<Self>
    {
        // Note, `PetscViewerASCIIGetStdout` calls `PetscObjectRegisterDestroy` which will cause
        // the object to be destroyed when `PetscFinalize()` is called. That is why we increase the 
        // reference count.
        let mut viewer_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_sys::PetscViewerASCIIGetStdout(world.as_raw(), viewer_p.as_mut_ptr()) };
        Petsc::check_error(world, ierr)?;

        let mut viewer = Viewer { world, viewer_p: unsafe { viewer_p.assume_init() } };
        unsafe { viewer.reference()?; }
        Ok(viewer)
    }

    /// Determines whether a PETSc [`Viewer`] is of a particular type.
    pub fn type_compare(&self, type_kind: ViewerType) -> Result<bool> {
        self.type_compare_str(&type_kind.to_string())
    }

    wrap_simple_petsc_member_funcs! {
        PetscViewerPushFormat, pub push_format, input PetscViewerFormat, format, takes mut, #[doc = "Sets the format for file PetscViewers."];
    }
}

impl_petsc_object_traits! { Viewer, viewer_p, petsc_raw::_p_PetscViewer }
