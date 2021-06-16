//! PETSc viewers export information and data from PETSc objects.
//!
//! PETSc C API docs: <https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Viewer/index.html>

use crate::prelude::*;

pub use petsc_sys::PetscViewerFormat;

/// Abstract collection of PetscViewers. It is just an expandable array of viewers.
// TODO: right now this is a very basic wrapper of the view functionality, I feel like
// we could do this more rusty, but i don't really know how.
// It might make sense to make our own viewer code. That we we can do everything on the rust side
// This would work more nicely with things like display and debug, also the file system.
// for now we will just use the C api.
pub struct Viewer<'a> {
    world: &'a dyn Communicator,

    pub(crate) viewer_p: *mut petsc_raw::_p_PetscViewer,
}

impl<'a> Drop for Viewer<'a> {
    fn drop(&mut self) {
        // TODO it is unclear when we call Destroy on a PetscViewer, but if we do we get an error.
        // unsafe {
        //     let ierr = petsc_raw::PetscViewerDestroy(&mut self.viewer_p as *mut *mut petsc_raw::_p_PetscViewer);
        //     let _ = self.petsc.check_error(ierr); // TODO should i unwrap or what idk?
        // }
    }
}

impl_petsc_object_funcs!{ Viewer, viewer_p }

impl<'a> Viewer<'a> {
    /// Creates a ASCII PetscViewer shared by all processors in a communicator.
    pub fn create_ascii_stdout(world: &'a dyn Communicator) -> Result<Self>
    {
        let mut viewer_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_sys::PetscViewerASCIIGetStdout(world.as_raw(), viewer_p.as_mut_ptr()) };
        Petsc::check_error(world, ierr)?;
        
        Ok(Viewer { world, viewer_p: unsafe { viewer_p.assume_init() } })
    }

    wrap_simple_petsc_member_funcs! {
        PetscViewerPushFormat, push_format, viewer_p, input PetscViewerFormat, format, takes mut, #[doc = "Sets the format for file PetscViewers."];
    }
}
