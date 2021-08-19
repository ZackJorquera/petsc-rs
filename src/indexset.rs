//! Index set (IS) objects are used to index into vectors and matrices and to setup vector scatters.
//!
//! PETSc C API docs: <https://petsc.org/release/docs/manualpages/IS/index.html>

use std::mem::MaybeUninit;
use std::ops::Deref;
use crate::{petsc_raw, Result, PetscAsRaw, PetscInt, PetscObject};
use mpi::topology::UserCommunicator;
use mpi::traits::*;

/// [`IS`] Type
pub use crate::petsc_raw::ISTypeEnum as ISType;

/// Abstract PETSc object that allows indexing. 
pub struct IS<'a> {
    pub(crate) world: &'a UserCommunicator,

    pub(crate) is_p: *mut petsc_raw::_p_IS,
}

/// A immutable view of the indices with Deref to slice.
pub struct ISView<'a, 'b> {
    is: &'b IS<'a>,
    array: *const PetscInt,
    // Or should this just be an ndarray
    // pub(crate) ndarray: ArrayView<'b, PetscInt, ndarray::Ix1>,
    slice: &'b [PetscInt],
}

impl Drop for ISView<'_, '_> {
    fn drop(&mut self) {
        let ierr = unsafe { petsc_raw::ISRestoreIndices(self.is.is_p, &mut self.array as *mut _) };
        let _ = unsafe { chkerrq!(self.is.world, ierr) }; // TODO: should I unwrap or what idk?
    }
}

impl<'a> IS<'a> {
    /// Creates an index set object. 
    pub fn create(world: &'a UserCommunicator) -> Result<Self> {
        let mut is_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::ISCreate(world.as_raw(), is_p.as_mut_ptr()) };
        unsafe { chkerrq!(world, ierr) }?;

        Ok(IS { world, is_p: unsafe { is_p.assume_init() } })
    }

    /// Returns [`ISView`] that derefs into a slice of the indices.
    pub fn get_indices(&self) -> Result<ISView<'a, '_>> {
        ISView::new(self)
    }

    /// Determines whether a PETSc [`IS`] is of a particular type.
    pub fn type_compare(&self, type_kind: ISType) -> Result<bool> {
        self.type_compare_str(&type_kind.to_string())
    }
}

impl<'a, 'b> ISView<'a, 'b> {
    /// Constructs a ISView from a IS reference
    fn new(is: &'b IS<'a>) -> Result<Self> {
        let mut array = MaybeUninit::<*const PetscInt>::uninit();
        let ierr = unsafe { petsc_raw::ISGetIndices(is.is_p, array.as_mut_ptr()) };
        unsafe { chkerrq!(is.world, ierr) }?;

        // let ndarray = unsafe { 
        //     ArrayView::from_shape_ptr(ndarray::Ix1(is.get_local_size()? as usize), array.assume_init()) };
        let slice = unsafe { 
            std::slice::from_raw_parts(array.assume_init(), is.get_local_size()? as usize)} ;

        // Ok(Self { is, array: unsafe { array.assume_init() }, ndarray })
        Ok(Self { is, array: unsafe { array.assume_init() }, slice })
    }
}

impl<'b> Deref for ISView<'_, 'b> {
    // type Target = ArrayView<'b, PetscInt, ndarray::Ix1>;
    // fn deref(&self) -> &ArrayView<'b, PetscInt, ndarray::Ix1> {
    //     &self.ndarray
    // }
    type Target = [PetscInt];
    fn deref(&self) -> &'b [PetscInt] {
        &self.slice
    }
}

impl std::fmt::Debug for ISView<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.slice.fmt(f)
    }
}

// macro impls
impl IS<'_> {
    wrap_simple_petsc_member_funcs! {
        ISGetLocalSize, pub get_local_size, output PetscInt, size, #[doc = "Returns the local (processor) length of an index set. "];
    }
}

impl_petsc_object_traits! { IS, is_p, petsc_raw::_p_IS, ISView, ISDestroy; }
