//! PETSc viewers export information and data from PETSc objects.
//!
//! PETSc C API docs: <https://petsc.org/release/docs/manualpages/Viewer/index.html>

use std::mem::MaybeUninit;
use crate::{
    petsc_raw,
    Result,
    PetscAsRaw,
    PetscObject,
    PetscObjectPrivate
};
use mpi::topology::UserCommunicator;
use mpi::traits::*;

/// Way a [`Viewer`] presents the object.
pub use petsc_sys::PetscViewerFormat;

/// [`Viewer`] Type
pub use crate::petsc_raw::ViewerTypeEnum as ViewerType;

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

impl<'a> Viewer<'a> {
    /// Creates a ASCII PetscViewer shared by all processors in a communicator.
    pub fn create_ascii_stdout(world: &'a UserCommunicator) -> Result<Self>
    {
        // Note, `PetscViewerASCIIGetStdout` calls `PetscObjectRegisterDestroy` which will cause
        // the object to be destroyed when `PetscFinalize()` is called. That is why we increase the 
        // reference count.
        let mut viewer_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_sys::PetscViewerASCIIGetStdout(world.as_raw(), viewer_p.as_mut_ptr()) };
        unsafe { chkerrq!(world, ierr) }?;

        let mut viewer = Viewer { world, viewer_p: unsafe { viewer_p.assume_init() } };
        unsafe { viewer.reference()?; }
        Ok(viewer)
    }

    /// Determines whether a PETSc [`Viewer`] is of a particular type.
    pub fn type_compare(&self, type_kind: ViewerType) -> Result<bool> {
        self.type_compare_str(&type_kind.to_string())
    }

    /// Views a [viewable PetscObject](PetscViewable).
    ///
    /// Same as [`obj.view_with(&self)`](PetscViewable::view_with())
    pub fn view<T: PetscViewable>(&self, obj: &T) -> Result<()> {
        obj.view_with(self)
    }

    wrap_simple_petsc_member_funcs! {
        PetscViewerPushFormat, pub push_format, input PetscViewerFormat, format, takes mut, #[doc = "Sets the format for file PetscViewers."];
    }
}

/// A PETSc object that can be viewed with a [`Viewer`]
pub trait PetscViewable {
    /// Views the object with a [`Viewer`]
    fn view_with<'vl, 'val: 'vl>(&self, viewer: impl Into<Option<&'vl Viewer<'val>>>) -> Result<()>;
}

impl_petsc_object_traits! { Viewer, viewer_p, petsc_raw::_p_PetscViewer, PetscViewerView, PetscViewerDestroy; }
