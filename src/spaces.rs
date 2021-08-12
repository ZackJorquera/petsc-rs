//! PETSc [`Space`] struct is used to encapsulate a function space.
//!
//! PETSc C API docs: <https://petsc.org/release/docs/manualpages/SPACE/index.html>

use std::{ffi::CString, mem::MaybeUninit};
use crate::{
    petsc_raw,
    Result,
    PetscAsRaw,
    PetscObject,
    PetscInt,
    dm::{DM, },
};
use mpi::topology::UserCommunicator;
use mpi::traits::*;

/// [`Space`] type
pub type SpaceType = petsc_raw::PetscSpaceTypeEnum;
/// [`DualSpace`] type
pub type DualSpaceType = petsc_raw::PetscDualSpaceTypeEnum;

/// Abstract PETSc object that encapsulates a function space.
pub struct Space<'a> {
    pub(crate) world: &'a UserCommunicator,
    pub(crate) s_p: *mut petsc_raw::_p_PetscSpace,
}

impl Drop for Space<'_> {
    fn drop(&mut self) {
        let ierr = unsafe { petsc_raw::PetscSpaceDestroy(&mut self.s_p as *mut _) };
        let _ = chkerrq!(self.world, ierr); // TODO: should I unwrap or what idk?
    }
}

/// PETSc object that manages the dual space to a linear space, e.g. the space of evaluation functionals at the vertices of a triangle 
pub struct DualSpace<'a, 'tl> {
    pub(crate) world: &'a UserCommunicator,
    pub(crate) ds_p: *mut petsc_raw::_p_PetscDualSpace,

    dm: Option<DM<'a, 'tl>>
}

impl Drop for DualSpace<'_, '_> {
    fn drop(&mut self) {
        let ierr = unsafe { petsc_raw::PetscDualSpaceDestroy(&mut self.ds_p as *mut _) };
        let _ = chkerrq!(self.world, ierr); // TODO: should I unwrap or what idk?
    }
}

impl<'a> Space<'a> {
    /// Creates an empty [`Space`] object.
    ///
    /// The type can then be set with [`Space::set_type()`]. 
    pub fn create(world: &'a UserCommunicator) -> Result<Self> {
        let mut s_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::PetscSpaceCreate(
            world.as_raw(), s_p.as_mut_ptr()) };
        chkerrq!(world, ierr)?;

        Ok(Space { world, s_p: unsafe { s_p.assume_init() } })
    }

    /// Builds a particular [`Space`] (given as `&str`).
    pub fn set_type_str(&mut self, space_type: &str) -> Result<()> {
        let cstring = CString::new(space_type).expect("`CString::new` failed");
        let ierr = unsafe { petsc_raw::PetscSpaceSetType(self.s_p, cstring.as_ptr()) };
        chkerrq!(self.world, ierr)
    }

    /// Builds a particular [`Space`].
    pub fn set_type(&mut self, space_type: SpaceType) -> Result<()> {
        // This could be use the macro probably 
        let option_str = petsc_raw::PETSCSPACETYPE_TABLE[space_type as usize];
        let ierr = unsafe { petsc_raw::PetscSpaceSetType(self.s_p, option_str.as_ptr() as *const _) };
        chkerrq!(self.world, ierr)
    }

    /// Determines whether a PETSc [`Space`] is of a particular type.
    pub fn type_compare(&self, type_kind: SpaceType) -> Result<bool> {
        self.type_compare_str(&type_kind.to_string())
    }

    /// Set the degree of approximation for this space.
    ///
    /// # Parameters
    ///
    /// * `degree` - The degree of the largest polynomial space contained in the space.
    /// * `max_degree` - The degree of the largest polynomial space containing the space.
    ///
    /// One of `degree` or `max_degree` can be `None` to have PETSc determine what it should be.
    pub fn set_degree(&mut self, degree: impl Into<Option<PetscInt>>, max_degree: impl Into<Option<PetscInt>>) -> Result<()> {
        let ierr = unsafe { petsc_raw::PetscSpaceSetDegree(
            self.as_raw(),
            degree.into().unwrap_or(petsc_raw::PETSC_DETERMINE),
            max_degree.into().unwrap_or(petsc_raw::PETSC_DETERMINE) 
        ) };
        chkerrq!(self.world, ierr)
    }
}

impl<'a, 'tl> DualSpace<'a, 'tl> {
    /// Creates an empty [`DualSpace`] object.
    ///
    /// The type can then be set with [`DualSpace::set_type()`]. 
    pub fn create(world: &'a UserCommunicator) -> Result<Self> {
        let mut ds_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::PetscDualSpaceCreate(
            world.as_raw(), ds_p.as_mut_ptr()) };
        chkerrq!(world, ierr)?;

        Ok(DualSpace { world, ds_p: unsafe { ds_p.assume_init() }, dm: None })
    }

    /// Builds a particular [`DualSpace`] (given as `&str`).
    pub fn set_type_str(&mut self, ds_type: &str) -> Result<()> {
        let cstring = CString::new(ds_type).expect("`CString::new` failed");
        let ierr = unsafe { petsc_raw::PetscDualSpaceSetType(self.ds_p, cstring.as_ptr()) };
        chkerrq!(self.world, ierr)
    }

    /// Builds a particular [`DualSpace`].
    pub fn set_type(&mut self, ds_type: DualSpaceType) -> Result<()> {
        // This could be use the macro probably 
        let option_str = petsc_raw::PETSCDUALSPACETYPE_TABLE[ds_type as usize];
        let ierr = unsafe { petsc_raw::PetscDualSpaceSetType(self.ds_p, option_str.as_ptr() as *const _) };
        chkerrq!(self.world, ierr)
    }

    /// Create a DMPLEX with the appropriate FEM reference cell 
    pub fn create_reference_cell(&self, dim: PetscInt, simplex: bool) -> Result<DM> {
        let mut dm_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::PetscDualSpaceCreateReferenceCell(self.ds_p,
            dim, simplex.into(), dm_p.as_mut_ptr()) };
        chkerrq!(self.world, ierr)?;

        Ok(DM::new(self.world, unsafe { dm_p.assume_init() }))
    }

    /// Determines whether a PETSc [`DualSpace`] is of a particular type.
    pub fn type_compare(&self, type_kind: DualSpaceType) -> Result<bool> {
        self.type_compare_str(&type_kind.to_string())
    }
}

// macro impls
impl<'a> Space<'a> {
    wrap_simple_petsc_member_funcs! {
        PetscSpaceSetFromOptions, pub set_from_options, takes mut, #[doc = "Sets parameters from the options database"];
        PetscSpaceSetUp, pub set_up, takes mut, #[doc = "Construct data structures"];
        PetscSpacePolynomialSetTensor, pub polynomial_set_tensor, input bool, tensor, takes mut, #[doc = "Set whether a function space is a space of tensor polynomials, as opposed to polynomials.\n\n\
            For tensor polynomials the space is spanned by polynomials whose degree in each variable is bounded by the given order. For polynomials the space is spanned by polynomials whose total degree---summing over all variables---is bounded by the given order."];
        PetscSpacePolynomialGetTensor, pub polynomial_get_tensor, output bool, tensor, #[doc = "Gets whether a function space is a space of tensor polynomials, as opposed to polynomials."];
        PetscSpaceSetNumComponents, pub set_num_components, input PetscInt, nc, takes mut, #[doc = "Set the number of components for this space"];
        PetscSpaceSetNumVariables, pub set_num_variables, input PetscInt, nv, takes mut, #[doc = "Set the number of variables for this space"];
        PetscSpaceGetNumComponents, pub get_num_components, output PetscInt, nc, #[doc = "Get the number of components for this space"];
        PetscSpaceGetNumVariables, pub get_num_variables, output PetscInt, nv, #[doc = "Get the number of variables for this space"];
        PetscSpaceGetDegree, pub get_degree, output PetscInt, degree, output PetscInt, max_degree, #[doc = "Get the degree of approximation for this space."];
    }
}

impl<'a, 'tl> DualSpace<'a, 'tl> {
    wrap_simple_petsc_member_funcs! {
        PetscDualSpaceSetFromOptions, pub set_from_options, takes mut, #[doc = "Sets parameters from the options database"];
        PetscDualSpaceSetUp, pub set_up, takes mut, #[doc = "Construct data structures"];
        PetscDualSpaceLagrangeSetTensor, pub lagrange_set_tensor, input bool, tensor, takes mut, #[doc = "Set the tensor nature of the dual space.\n\n\
            Whether the dual space has tensor layout (vs. simplicial)"];
        PetscDualSpaceLagrangeGetTensor, pub lagrange_get_tensor, output bool, tensor, #[doc = "Gets whether the dual space has tensor layout (vs. simplicial)"];
        PetscDualSpaceSetNumComponents, pub set_num_components, input PetscInt, nc, takes mut, #[doc = "Set the number of components for this space"];
        PetscDualSpaceSetOrder, pub set_order, input PetscInt, degree, takes mut, #[doc = "Set the order of the dual space."];
        PetscDualSpaceGetNumComponents, pub get_num_components, output PetscInt, nc, #[doc = "Get the number of components for this space"];
        PetscDualSpaceGetOrder, pub get_order, output PetscInt, degree, #[doc = "Get the order of the dual space."];
        PetscDualSpaceSetDM, pub set_dm, input DM<'a, 'tl>, dm .as_raw consume .dm, takes mut, #[doc = "Set the DM representing the reference cell "];
    }
}

impl_petsc_object_traits! {
    Space, s_p, petsc_raw::_p_PetscSpace, PetscSpaceView;
    DualSpace, ds_p, petsc_raw::_p_PetscDualSpace, PetscDualSpaceView, '_;
}
