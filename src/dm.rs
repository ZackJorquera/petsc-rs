//! DM objects are used to manage communication between the algebraic structures in PETSc ([`Vector`] and [`Mat`])
//! and mesh data structures in PDE-based (or other) simulations. See, for example, [`DM::da_create_1d()`].
//!
//! The DMDA class encapsulates a Cartesian structured mesh, with interfaces for both topology and geometry.
//! It is capable of parallel refinement and coarsening. Some support for parallel redistribution is
//! available through the PCTELESCOPE object. A piecewise linear discretization is assumed for operations
//! which require this information.
//!
//! PETSc C API docs: <https://petsc.org/release/docs/manualpages/DM/index.html>
//!
//! Also: <https://petsc.org/release/docs/manualpages/DMDA/DMDA.html#DMDA>

// TODO: use `PetscObjectTypeCompare` (`DM::type_compare`) to make sure we are using the correct type of DM
// for different functions.

use core::slice;
use std::marker::PhantomData;
use std::mem::{MaybeUninit, ManuallyDrop};
use std::ffi::{CString, CStr};
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::pin::Pin;
use std::ptr::NonNull;
use std::rc::Rc;
use crate::{
    Petsc,
    petsc_raw,
    Result,
    PetscAsRaw,
    PetscObject,
    PetscObjectPrivate,
    PetscReal,
    PetscInt,
    PetscScalar,
    PetscErrorKind,
    InsertMode,
    vector::{self, Vector, },
    mat::{Mat, MatType, },
    indexset::{IS, },
    spaces::{Space, DualSpace, },
    snes::{SNES, DomainOrPetscError, },
    ksp::{KSP, },
    petsc_panic,
};
use mpi::topology::UserCommunicator;
use mpi::traits::*;

use ndarray::{ArrayView, ArrayViewMut};

/// Abstract PETSc object that manages an abstract grid object and its interactions with the algebraic solvers
pub struct DM<'a, 'tl> {
    pub(crate) world: &'a UserCommunicator,
    pub(crate) dm_p: *mut petsc_raw::_p_DM,

    composite_dms: Option<Vec<DM<'a, 'tl>>>,

    coord_dm: Option<Rc<DM<'a, 'tl>>>,
    coarse_dm: Option<Box<DM<'a, 'tl>>>,

    fields: Option<Vec<(Option<DMLabel<'a>>, FieldDiscPriv<'a, 'tl>)>>,

    ds: Option<DS<'a, 'tl>>,

    #[allow(dead_code)]
    aux_vec: Option<Vector<'a>>,

    snes_function_trampoline_data: Option<Pin<Box<SNESFunctionTrampolineData<'a, 'tl>>>>,
    snes_jacobian_trampoline_data: Option<SNESJacobianTrampolineData<'a, 'tl>>,

    ksp_compute_operators_trampoline_data: Option<Pin<Box<KSPComputeOperatorsTrampolineData<'a, 'tl>>>>,
    ksp_compute_rhs_trampoline_data: Option<Pin<Box<KSPComputeRHSTrampolineData<'a, 'tl>>>>,
}

/// Object which encapsulates a subset of the mesh from a [`DM`]
pub struct DMLabel<'a> {
    pub(crate) world: &'a UserCommunicator,
    pub(crate) dml_p: *mut petsc_raw::_p_DMLabel,
}

/// The [`FEDisc`] class encapsulates a finite element discretization.
///
/// Each [`FEDisc`] object contains a [`Space`], [`DualSpace`], and
/// DMPlex in the classic Ciarlet triple representation. 
pub struct FEDisc<'a, 'tl> {
    pub(crate) world: &'a UserCommunicator,
    pub(crate) fe_p: *mut petsc_raw::_p_PetscFE,

    space: Option<Space<'a>>,
    dual_space: Option<DualSpace<'a, 'tl>>,
    // TODO: add dm
}

/// The [`FVDisc`] class encapsulates a finite volume discretization.
///
/// TODO: implement
pub struct FVDisc<'a> {
    pub(crate) world: &'a UserCommunicator,
    pub(crate) fv_p: *mut petsc_raw::_p_PetscFV,
}

/// PETSc object for defining a field on a mesh topology
///
/// TODO: implement
pub struct DMField<'a> {
    pub(crate) world: &'a UserCommunicator,
    pub(crate) field_p: *mut petsc_raw::_p_DMField,
}

/// A enum that can represents types of discretization objects.
///
/// Many C API function that take a field, take it as a `PetscObject` to allow for
/// multiple types of fields. This trait acts in the same way. An example of this trait
/// being used is the method [`DM::add_field()`].
///
/// All variants implement `Into<Field>` so it's easier to use.
pub enum FieldDisc<'a, 'tl> {
    // From what i can tell these are the only two options:
    /// The discretization object is [`FEDisc`]
    FEDisc(FEDisc<'a, 'tl>),
    /// The discretization object is [`FVDisc`]
    FVDisc(FVDisc<'a>),
}

/// internal type used only for storage.
// TODO: we could get rid of this struct by checking what type the FieldDisc is
// by using `PetscObjectGetClassId` like is done here (look at `into_known_unwrap`):
// https://gitlab.com/petsc/petsc/-/blob/9634419/src/dm/impls/plex/plexsection.c#L454-456
enum FieldDiscPriv<'a, 'tl> {
    Known(FieldDisc<'a, 'tl>),
    /// The discretization object is unknown or we dont care what it is,
    /// this is only used internally.
    Unknown(crate::PetscObjectStruct<'a>),
}

unsafe impl PetscAsRaw for FieldDisc<'_, '_> {
    type Raw = *mut petsc_raw::_p_PetscObject;

    #[inline]
    fn as_raw(&self) -> Self::Raw {
        match self {
            FieldDisc::FEDisc(f) => f.as_raw() as *mut _,
            FieldDisc::FVDisc(f) => f.as_raw() as *mut _,
        }
    }
}

unsafe impl<'a> crate::PetscAsRawMut for FieldDisc<'_, '_> {
    #[inline]
    fn as_raw_mut(&mut self) -> *mut Self::Raw {
        match self {
            FieldDisc::FEDisc(f) => &mut f.as_raw() as *mut *mut _ as *mut _,
            FieldDisc::FVDisc(f) => &mut f.as_raw() as *mut *mut _ as *mut _,
        }
    }
} 

impl<'a> crate::PetscObject<'a, petsc_raw::_p_PetscObject> for FieldDisc<'a, '_> {
    #[inline]
    fn world(&self) -> &'a mpi::topology::UserCommunicator {
        match self {
            FieldDisc::FEDisc(f) => f.world(),
            FieldDisc::FVDisc(f) => f.world(),
        }
    }
}

impl<'a> crate::PetscObjectPrivate<'a, petsc_raw::_p_PetscObject> for FieldDisc<'a, '_> { }

impl<'a, 'tl> Into<FieldDisc<'a, 'tl>> for FEDisc<'a, 'tl> {
    fn into(self) -> FieldDisc<'a, 'tl> {
        FieldDisc::FEDisc(self)
    }
}

impl<'a, 'tl> Into<FieldDisc<'a, 'tl>> for FVDisc<'a> {
    fn into(self) -> FieldDisc<'a, 'tl> {
        FieldDisc::FVDisc(self)
    }
}

impl<'a, 'tl> FieldDiscPriv<'a, 'tl> {
    #[allow(dead_code)]
    fn into_known_unwrap(self) -> FieldDisc<'a, 'tl> {
        match self {
            FieldDiscPriv::Known(fd) => fd,
            FieldDiscPriv::Unknown(ufd) => {
                let id = Petsc::unwrap_or_abort(ufd.get_class_id(), ufd.world());
                if id == unsafe { petsc_raw::PETSCFE_CLASSID } {
                    FEDisc::new(ufd.world, ufd.po_p as *mut _).into()
                } else if id == unsafe { petsc_raw::PETSCFV_CLASSID } {
                    FVDisc::new(ufd.world, ufd.po_p as *mut _).into()
                } else {
                    petsc_panic!(ufd.world, PetscErrorKind::PETSC_ERR_ARG_WRONG, 
                        "Unknown discretization type for field")
                }
            }
        }
    }
}

/// PETSc object that manages a discrete system, which is a set of
/// discretizations + continuum equations from a [`WeakForm`].
pub struct DS<'a, 'tl> {
    pub(crate) world: &'a UserCommunicator,
    pub(crate) ds_p: *mut petsc_raw::_p_PetscDS,

    #[allow(dead_code)]
    residual_trampoline_data: Option<DSResidualTrampolineData<'tl>>,
    #[allow(dead_code)]
    jacobian_trampoline_data: Option<DSJacobianTrampolineData<'tl>>,

    exact_soln_trampoline_data: Option<Pin<Box<DSExactSolutionTrampolineData<'tl>>>>,

    // TODO: we need to do a lot more work on determining how to clone a DM and thus
    // also a DS on the rust side.
    // Idea: We could reference count the trampoline data because under the hood
    // the DMPLEX data is shallow copied and reference counted.
    // Even if this isn't copied when the DM/DS is cloned, it probably wont matter in the short
    // term as we never use old boundary data (unless this comment is out of date and we do).
    // Right now, we just create the boundary data, give the pointer to C api and never use
    // it again util it is dropped, so having the Rc just prolongs the time until drop.
    // However, if this is the case we should remove the `Rc` as it could be unsafe.
    boundary_trampoline_data: Option<Vec<DMBoundaryTrampolineData<'tl>>>,
}

/// PETSc object that manages a sets of pointwise functions defining a system of equations 
pub struct WeakForm<'a> {
    pub(crate) world: &'a UserCommunicator,
    pub(crate) wf_p: *mut petsc_raw::_p_PetscWeakForm,
}

// TODO: is this even needed?
/// A wrapper around [`DM`] that is used when the inner [`DM`] might contain closures from another [`DM`].
///
/// For example, it is used with [`DM::clone_shallow()`].
pub struct BorrowDM<'a, 'tl, 'bv> {
    // for now we will just use a normal drop function.
    owned_dm: DM<'a, 'tl>,
    // drop_func: Option<Box<dyn FnOnce(&mut Self) + 'bv>>,
    pub(crate) _phantom: PhantomData<&'bv mut DM<'a, 'tl>>,
}

/// Describes the choice for fill of ghost cells on physical domain boundaries.
///
/// <https://petsc.org/release/docs/manualpages/DM/DMBoundaryType.html>
pub use crate::petsc_raw::DMBoundaryType;
/// Determines if the stencil extends only along the coordinate directions, or also to the northeast, northwest etc.
pub use crate::petsc_raw::DMDAStencilType;
/// [`DM`] Type
pub use crate::petsc_raw::DMTypeEnum as DMType;
/// Indicates what type of boundary condition is to be imposed
///
/// <https://petsc.org/release/docs/manualpages/DM/DMBoundaryConditionType.html>
pub use crate::petsc_raw::DMBoundaryConditionType;

/// [`DMField`] Type
pub use crate::petsc_raw::DMFieldTypeEnum as DMFieldType;
/// [`FEDisc`] Type
pub use crate::petsc_raw::PetscFETypeEnum as FEDiscType;
/// [`FVDisc`] Type
pub use crate::petsc_raw::PetscFVTypeEnum as FVDiscType;

enum DMBoundaryTrampolineData<'tl> {
    BCFunc(Pin<Box<DMBoundaryFuncTrampolineData<'tl>>>),
    #[allow(dead_code)]
    BCField(Pin<Box<DMBoundaryFieldTrampolineData<'tl>>>),
}

struct DMBoundaryFuncTrampolineData<'tl> {
    user_f1: Box<BCFuncDyn<'tl>>,
    user_f2: Option<Box<BCFuncDyn<'tl>>>,
}

#[allow(dead_code)]
struct DMBoundaryFieldTrampolineData<'tl> {
    user_f1: Box<BCFieldDyn<'tl>>,
    user_f2: Option<Box<BCFieldDyn<'tl>>>,
}

// TODO: make use the real function stuff
#[allow(dead_code)]
struct DSResidualTrampolineData<'tl> {
    user_f0: Option<Box<dyn Fn() + 'tl>>,
    user_f1: Option<Box<dyn Fn() + 'tl>>,
}

#[allow(dead_code)]
struct DSJacobianTrampolineData<'tl> {
    user_f0: Option<Box<dyn Fn() + 'tl>>,
    user_f1: Option<Box<dyn Fn() + 'tl>>,
}

struct DSExactSolutionTrampolineData<'tl> {
    user_f: Box<BCFuncDyn<'tl>>,
}

struct DMProjectFunctionTrampolineData<'tl> {
    user_f: Box<dyn FnMut(PetscInt, PetscReal, &[PetscReal],
    PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl>,
}

// TODO: should i use trait aliases. It doesn't really matter, but the Fn types are long
type BCFuncDyn<'tl> = dyn Fn(PetscInt, PetscReal, &[PetscReal],
    PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl;
// Because the C method that uses this method doesn't have a context, we cant use this
type BCFieldDyn<'tl> = dyn Fn(PetscInt, PetscInt, PetscInt,
    &[PetscInt], &[PetscInt], &[PetscReal], &[PetscReal], &[PetscReal],
    &[PetscInt], &[PetscInt], &[PetscReal], &[PetscReal], &[PetscReal],
    PetscReal, &[PetscReal], PetscInt, &[PetscScalar], &mut [PetscScalar]) -> Result<()> + 'tl;

struct SNESFunctionTrampolineData<'a, 'tl> {
    world: &'a UserCommunicator,
    user_f: Box<dyn FnMut(&SNES<'a, '_, '_>, &Vector<'a>, &mut Vector<'a>) -> std::result::Result<(), DomainOrPetscError> + 'tl>,
}

enum SNESJacobianTrampolineData<'a, 'tl> {
    SingleMat(Pin<Box<SNESJacobianSingleTrampolineData<'a, 'tl>>>),
    DoubleMat(Pin<Box<SNESJacobianDoubleTrampolineData<'a, 'tl>>>),
}

struct SNESJacobianSingleTrampolineData<'a, 'tl> {
    world: &'a UserCommunicator,
    user_f: Box<dyn FnMut(&SNES<'a, '_, '_>, &Vector<'a>, &mut Mat<'a, '_>) -> std::result::Result<(), DomainOrPetscError> + 'tl>,
}

struct SNESJacobianDoubleTrampolineData<'a, 'tl> {
    world: &'a UserCommunicator,
    user_f: Box<dyn FnMut(&SNES<'a, '_, '_>, &Vector<'a>, &mut Mat<'a, '_>, &mut Mat<'a, '_>) -> std::result::Result<(), DomainOrPetscError> + 'tl>,
}

struct KSPComputeOperatorsTrampolineData<'a, 'tl> {
    world: &'a UserCommunicator,
    user_f: Box<dyn FnMut(&KSP<'a, '_, '_>, &mut Mat<'a, '_>, &mut Mat<'a, '_>) -> Result<()> + 'tl>,
}

struct KSPComputeRHSTrampolineData<'a, 'tl> {
    world: &'a UserCommunicator,
    user_f: Box<dyn FnMut(&KSP<'a, '_, '_>, &mut Vector<'a>) -> Result<()> + 'tl>,
}

impl<'a, 'tl> DM<'a, 'tl> {
    /// Same as `DM { ... }` but sets all optional params to `None`
    pub(crate) fn new(world: &'a UserCommunicator, dm_p: *mut petsc_raw::_p_DM) -> Self {
        DM { world, dm_p, composite_dms: None, fields: None,
            ds: None, coord_dm: None, coarse_dm: None, aux_vec: None,
            snes_function_trampoline_data: None,
            snes_jacobian_trampoline_data: None,
            ksp_compute_operators_trampoline_data: None,
            ksp_compute_rhs_trampoline_data: None, }
    }

    /// Creates an empty [`DM`] object. The type can then be set with [`DM::set_type()`].
    pub fn create(world: &'a UserCommunicator) -> Result<Self> {
        let mut dm_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::DMCreate(world.as_raw(), dm_p.as_mut_ptr()) };
        unsafe { chkerrq!(world, ierr) }?;

        Ok(DM::new(world, unsafe { dm_p.assume_init() }))
    }

    /// Builds [`DM`] for a particular DM implementation (given as `&str`).
    pub fn set_type_str(&mut self, dm_type: &str) -> Result<()> {
        let cstring = CString::new(dm_type).expect("`CString::new` failed");
        let ierr = unsafe { petsc_raw::DMSetType(self.dm_p, cstring.as_ptr()) };
        unsafe { chkerrq!(self.world, ierr) }
    }

    /// Builds a [`DM`], for a particular DM implementation.
    pub fn set_type(&mut self, dm_type: DMType) -> Result<()> {
        // This could be use the macro probably 
        let option_cstr = petsc_raw::DMTYPE_TABLE[dm_type as usize];
        let ierr = unsafe { petsc_raw::DMSetType(self.dm_p, option_cstr.as_ptr() as *const _) };
        unsafe { chkerrq!(self.world, ierr) }
    }

    /// Sets the type of [`Mat`] returned by [`DM::create_matrix()`].
    pub fn set_mat_type(&mut self, mat_type: MatType) -> Result<()> {
        let type_name_cs = ::std::ffi::CString::new(mat_type.to_string()).expect("`CString::new` failed");
        let ierr = unsafe { petsc_raw::DMSetMatType(self.dm_p, type_name_cs.as_ptr()) };
        unsafe { chkerrq!(self.world, ierr) }
    }

    /// Creates an object that will manage the communication of one-dimensional regular array data
    /// that is distributed across some processors.
    ///
    /// # Parameters
    /// * `world` - MPI communicator world
    /// * `bx` - type of ghost cells at the boundary the array should have, if any
    /// * `nx` - global dimension of the array (that is the number of grid points)
    /// * `dof` - number of degrees of freedom per node
    /// * `s` - stencil width
    /// * `lx` _(optional)_ - array containing number of nodes in the x direction on each processor, or `None`.
    /// If `Some(...)`, must be of length as the number of processes in the MPI world (i.e. `world.size()`).
    /// The sum of these entries must equal `nx`.
    pub fn da_create_1d<'ll>(world: &'a UserCommunicator, bx: DMBoundaryType, nx: PetscInt, dof: PetscInt,
        s: PetscInt, lx: impl Into<Option<&'ll [PetscInt]>>) -> Result<Self>
    {
        let lx = lx.into();
        assert!(lx.map_or(true, |lx| lx.len() == world.size() as usize));
        let mut dm_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::DMDACreate1d(world.as_raw(), bx, nx,
            dof, s, lx.map_or(std::ptr::null(), |lx| lx.as_ptr()), dm_p.as_mut_ptr()) };
        unsafe { chkerrq!(world, ierr) }?;

        Ok(DM::new(world, unsafe { dm_p.assume_init() }))
    }

    /// Creates an object that will manage the communication of two-dimensional regular array data
    /// that is distributed across some processors.
    ///
    /// # Parameters
    /// * `world` - MPI communicator world
    /// * `bx, by` - type of ghost cells at the boundary the array should have, if any
    /// * `stencil_type` - stencil type
    /// * `nx, ny` - global dimension in each direction of the array (that is the number of grid points)
    /// * `px, py` - corresponding number of processors in each dimension (or `None` to have calculated).
    /// * `dof` - number of degrees of freedom per node
    /// * `s` - stencil width
    /// * `lx, ly` _(optional)_ - arrays containing the number of nodes in each cell along the x and y
    /// coordinates, or `None`. If `Some(...)`, these must the same length as `px` and `py`, and the
    /// corresponding `px` and `py` cannot be `None`. The sum of the `lx` entries must be `nx`, and
    /// the sum of the `ly` entries must be `ny`.
    pub fn da_create_2d<'ll1, 'll2>(world: &'a UserCommunicator, bx: DMBoundaryType, by: DMBoundaryType, stencil_type: DMDAStencilType, 
        nx: PetscInt, ny: PetscInt, px: impl Into<Option<PetscInt>>, py: impl Into<Option<PetscInt>>, dof: PetscInt, s: PetscInt,
        lx: impl Into<Option<&'ll1 [PetscInt]>>, ly: impl Into<Option<&'ll2 [PetscInt]>>) -> Result<Self>
    {
        let lx = lx.into();
        let ly = ly.into();
        let px = px.into().unwrap_or(petsc_raw::PETSC_DECIDE_INTEGER);
        let py = py.into().unwrap_or(petsc_raw::PETSC_DECIDE_INTEGER);
        assert!(lx.map_or(true, |lx| lx.len() as PetscInt == px));
        assert!(ly.map_or(true, |ly| ly.len() as PetscInt == py));

        let mut dm_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::DMDACreate2d(world.as_raw(), bx, by, stencil_type, nx, ny, px, py,
            dof, s, lx.map_or(std::ptr::null(), |lx| lx.as_ptr()), ly.map_or(std::ptr::null(), |ly| ly.as_ptr()), dm_p.as_mut_ptr()) };
        unsafe { chkerrq!(world, ierr) }?;

        Ok(DM::new(world, unsafe { dm_p.assume_init() }))
    }

    /// Creates an object that will manage the communication of three-dimensional regular array data
    /// that is distributed across some processors.
    ///
    /// # Parameters
    /// * `world` - MPI communicator world
    /// * `bx, by, bz` - type of ghost cells at the boundary the array should have, if any
    /// * `stencil_type` - stencil type
    /// * `nx, ny, nz` - global dimension in each direction of the array (that is the number of grid points)
    /// * `px, py, pz` - corresponding number of processors in each dimension (or `None` to have calculated).
    /// * `dof` - number of degrees of freedom per node
    /// * `s` - stencil width
    /// * `lx, ly, lz` _(optional)_ - arrays containing the number of nodes in each cell along the x, y, and z
    /// coordinates, or `None`. If `Some(...)`, these must the same length as `px`, `py`, and `pz`, and the
    /// corresponding `px`, `py`, and `pz` cannot be `None`. The sum of the `lx` entries must be `nx`,
    /// the sum of the `ly` entries must be `ny`, and the sum of the `lz` entries must be `nz`.
    pub fn da_create_3d<'ll1, 'll2, 'll3>(world: &'a UserCommunicator, bx: DMBoundaryType, by: DMBoundaryType, bz: DMBoundaryType, stencil_type: DMDAStencilType, 
        nx: PetscInt, ny: PetscInt, nz: PetscInt, px: impl Into<Option<PetscInt>>, py: impl Into<Option<PetscInt>>, pz: impl Into<Option<PetscInt>>, dof: PetscInt, s: PetscInt,
        lx: impl Into<Option<&'ll1 [PetscInt]>>, ly: impl Into<Option<&'ll2 [PetscInt]>>, lz: impl Into<Option<&'ll3 [PetscInt]>>) -> Result<Self>
    {
        let lx = lx.into();
        let ly = ly.into();
        let lz = lz.into();
        let px = px.into().unwrap_or(petsc_raw::PETSC_DECIDE_INTEGER);
        let py = py.into().unwrap_or(petsc_raw::PETSC_DECIDE_INTEGER);
        let pz = pz.into().unwrap_or(petsc_raw::PETSC_DECIDE_INTEGER);
        assert!(lx.map_or(true, |lx| lx.len() as PetscInt == px));
        assert!(ly.map_or(true, |ly| ly.len() as PetscInt == py));
        assert!(lz.map_or(true, |lz| lz.len() as PetscInt == pz));

        let mut dm_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::DMDACreate3d(world.as_raw(), bx, by, bz, stencil_type, nx, ny, nz, px, py, pz,
            dof, s, lx.map_or(std::ptr::null(), |lx| lx.as_ptr()), ly.map_or(std::ptr::null(), |ly| ly.as_ptr()),
            lz.map_or(std::ptr::null(), |lz| lz.as_ptr()), dm_p.as_mut_ptr()) };
        unsafe { chkerrq!(world, ierr) }?;

        Ok(DM::new(world, unsafe { dm_p.assume_init() }))
    }

    /// Creates a vector packer, used to generate "composite" vectors made up of several subvectors.
    pub fn composite_create<I>(world: &'a UserCommunicator, dms: I) -> Result<Self>
    where
        I: IntoIterator<Item = Self>
    {
        let mut dm_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::DMCompositeCreate(world.as_raw(), dm_p.as_mut_ptr()) };
        unsafe { chkerrq!(world, ierr) }?;

        let mut dm = DM::new(world, unsafe { dm_p.assume_init() });

        let dms = dms.into_iter().collect::<Vec<_>>();
        let _num_dm = dms.iter().map(|this_dm| {
            let ierr = unsafe { petsc_raw::DMCompositeAddDM(dm.dm_p, this_dm.dm_p) };
            unsafe { chkerrq!(dm.world, ierr) }.map(|_| 1)
        }).sum::<Result<PetscInt>>()?;

        dm.composite_dms = Some(dms);

        Ok(dm)
    }

    /// Creates a DMPlex object, which encapsulates an unstructured mesh,
    /// or CW complex, which can be expressed using a Hasse Diagram. 
    pub fn plex_create(world: &'a UserCommunicator) -> Result<Self> {
        let mut dm_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::DMPlexCreate(world.as_raw(), dm_p.as_mut_ptr()) };
        unsafe { chkerrq!(world, ierr) }?;

        Ok(DM::new(world, unsafe { dm_p.assume_init() }))
    }

    /// Creates a mesh on the tensor product of unit intervals (box) using simplices or
    /// tensor cells (hexahedra). 
    ///
    /// Also read: <https://petsc.org/release/docs/manualpages/DMPLEX/DMPlexCreateBoxMesh.html>
    ///
    /// # Parameters
    ///
    /// * `comm`        - The communicator for the DM object
    /// * `dim`         - The spatial dimension
    /// * `simplex`     - `true` for simplices, `false` for tensor cells
    /// * `faces`       - Number of faces per dimension, or `None` for `(1,)` in 1D, `(2, 2,)` in 2D
    /// and `(1, 1, 1)` in 3D. Note, if `dim` is less than 3 than the unneeded values are ignored.
    /// * `lower`       - The lower left corner, or `None` for `(0, 0, 0)`. Note, if `dim` is less
    /// than 3 than the unneeded values are ignored.
    /// * `upper`       - The upper right corner, or `None` for `(1, 1, 1)`. Note, if `dim` is less
    /// than 3 than the unneeded values are ignored.
    /// * `periodicity` - The boundary type for the X,Y,Z direction, or `None` for [`DM_BOUNDARY_NONE`](petsc_raw::DMBoundaryType::DM_BOUNDARY_NONE)
    /// for all directions. Note, if `dim` is less than 3 than the unneeded values are ignored.
    /// * `interpolate` - Flag to create intermediate mesh pieces (edges, faces)
    ///
    /// # Example
    ///
    /// Look at example for [`DM::project_function_local()`].
    pub fn plex_create_box_mesh(world: &'a UserCommunicator, dim: PetscInt, simplex: bool,
        faces: impl Into<Option<(PetscInt, PetscInt, PetscInt)>>,
        lower: impl Into<Option<(PetscReal, PetscReal, PetscReal)>>,
        upper: impl Into<Option<(PetscReal, PetscReal, PetscReal)>>,
        periodicity: impl Into<Option<(DMBoundaryType, DMBoundaryType, DMBoundaryType)>>,
        interpolate: bool) -> Result<Self>
    {
        let faces = faces.into().map(|faces| [faces.0, faces.1, faces.2]);
        let lower = lower.into().map(|lower| [lower.0, lower.1, lower.2]);
        let upper = upper.into().map(|upper| [upper.0, upper.1, upper.2]);
        let periodicity = periodicity.into().map(|periodicity| vec![periodicity.0, periodicity.1, periodicity.2]);

        let mut dm_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::DMPlexCreateBoxMesh(world.as_raw(), dim, simplex.into(),
            faces.as_ref().map_or(std::ptr::null(), |f| f.as_ptr()), lower.as_ref().map_or(std::ptr::null(), |l| l.as_ptr()),
            upper.as_ref().map_or(std::ptr::null(), |u| u.as_ptr()), periodicity.as_ref().map_or(std::ptr::null(), |p| p.as_ptr()),
            interpolate.into(), dm_p.as_mut_ptr()) };
        unsafe { chkerrq!(world, ierr) }?;

        Ok(DM::new(world, unsafe { dm_p.assume_init() }))
    }

    /// Creates a [`DM`] from a mesh file.
    pub fn plex_create_from_file(world: &'a UserCommunicator, file_name: impl AsRef<Path>, interpolate: bool) -> Result<Self> {
        let filename_cs = CString::new(file_name.as_ref().to_str().expect("`Path::to_str` failed")).expect("`CString::new` failed");
        let mut dm_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::DMPlexCreateFromFile(world.as_raw(), filename_cs.as_ptr(),
            interpolate.into(), dm_p.as_mut_ptr()) };
        unsafe { chkerrq!(world, ierr) }?;

        Ok(DM::new(world, unsafe { dm_p.assume_init() }))
    }

    /// Creates a global vector from a DM object
    pub fn create_global_vector(&self) -> Result<Vector<'a>> {
        let mut vec_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::DMCreateGlobalVector(self.dm_p, vec_p.as_mut_ptr()) };
        unsafe { chkerrq!(self.world, ierr) }?;
        
        Ok(Vector { world: self.world, vec_p: unsafe { vec_p.assume_init() } })
    }

    // TODO: Is the world usage correct? Like do we use the
    // world of the DM or just a single processes from it? or does it not matter?
    /// Creates a local vector from a DM object
    pub fn create_local_vector(&self) -> Result<Vector<'a>> {
        let mut vec_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::DMCreateLocalVector(self.dm_p, vec_p.as_mut_ptr()) };
        unsafe { chkerrq!(self.world, ierr) }?;
        
        Ok(Vector { world: self.world, vec_p: unsafe { vec_p.assume_init() } })
    }

    /// Gets a PETSc vector that may be used with the DM local routines.
    ///
    /// This vector has spaces for the ghost values. 
    ///
    /// The vector values are NOT initialized and may have garbage in them, so you may need to zero them.
    /// This is intended to be used if you need a vector for a short time, like within a single function
    /// call. For vectors that you intend to keep around (for example in a C struct) or pass around large
    /// parts of your code you should use [`DM::create_local_vector()`]. 
    pub fn get_local_vector(&self) -> Result<vector::BorrowVectorMut<'a, '_>> {
        // Note, under the hood `DMGetLocalVector` uses multiple different work vector 
        // that it will give access to us. Once it runs out it starts using `DMCreateLocalVector`.
        // Therefor we don't need to worry about this being called multiple times and causing
        // problems. At least I think we don't (it feels like interior mutability so i wonder if
        // we should be using UnsafeCell for something).

        let mut vec_p = MaybeUninit::uninit();
        // TODO: under the hood this does mutate the C struct but we dont take mut ref, is that a problem?
        // I dont think it is as the borrow rules are followed, i.e., we could theoretically used a `RefCell`
        // around `localin` and `localout` (the two arrays mutated), and we would be fine. Although, there
        // is interior mutability with out an `UnsafeCell` and idk if that will cause problems even if it is
        // safe.
        let ierr = unsafe { petsc_raw::DMGetLocalVector(self.dm_p, vec_p.as_mut_ptr()) };
        unsafe { chkerrq!(self.world, ierr) }?;

        // We dont want to drop the vec through Vector::drop
        let vec = ManuallyDrop::new(Vector { world: self.world, vec_p: unsafe { vec_p.assume_init() } });
        
        Ok(vector::BorrowVectorMut::new(vec, Some(Box::new(move |borrow_vec| {
            let ierr = unsafe { petsc_raw::DMRestoreLocalVector(self.dm_p, &mut borrow_vec.vec_p as *mut _) };
            let _ = unsafe { chkerrq!(borrow_vec.world, ierr) }; // TODO: should I unwrap ?
        }))))
    }

    /// Creates local vectors for each part of a DMComposite.
    ///
    /// Calls [`DM::create_local_vector()`] on each dm in the composite dm.
    pub fn composite_create_local_vectors(&self) -> Result<Vec<Vector<'a>>> {
        if let Some(c) = self.composite_dms.as_ref() {
            c.iter().fold(Ok(Vec::with_capacity(c.len())), |acc, dm| {
                match acc {
                    Ok(mut inner_vec) => dm.create_local_vector().map(|v| {
                            inner_vec.push(v);
                            inner_vec
                        }),
                    acc @ Err(_) => acc
                }
            })
        } else {
            seterrq!(self.world, PetscErrorKind::PETSC_ERR_ARG_WRONGSTATE,
                format!("There are no composite dms set, line: {}", line!())).map(|_| unreachable!())
        }
    }

    /// Gets local vectors for each part of a DMComposite.
    ///
    /// Calls [`DM::get_local_vector()`] on each dm in the composite dm.
    pub fn composite_get_local_vectors(&self) -> Result<Vec<vector::BorrowVectorMut<'a, '_>>> {
        if let Some(c) = self.composite_dms.as_ref() {
            c.iter().fold(Ok(Vec::with_capacity(c.len())), |acc, dm| {
                match acc {
                    Ok(mut inner_vec) => dm.get_local_vector().map(|v| {
                            inner_vec.push(v);
                            inner_vec
                        }),
                    acc @ Err(_) => acc
                }
            })
        } else {
            seterrq!(self.world, PetscErrorKind::PETSC_ERR_ARG_WRONGSTATE,
                format!("There are no composite dms set, line: {}", line!())).map(|_| unreachable!())
        }
    }

    /// Gets empty Jacobian for a DM
    ///
    /// This properly preallocates the number of nonzeros in the sparse
    /// matrix so you do not need to do it yourself. 
    ///
    /// For structured grid problems, when you call [`view_with()`](crate::viewer::PetscViewable::view_with()) on this matrix it is
    /// displayed using the global natural ordering, NOT in the ordering used internally by PETSc.
    pub fn create_matrix(&self) -> Result<Mat<'a, 'tl>> {
        let mut mat_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::DMCreateMatrix(self.dm_p, mat_p.as_mut_ptr()) };
        unsafe { chkerrq!(self.world, ierr) }?;
        
        Ok(Mat::new(self.world, unsafe { mat_p.assume_init() }))
    }

    /// Updates global vectors from local vectors.
    pub fn global_to_local(&self, global: &Vector, mode: InsertMode, local: &mut Vector) -> Result<()> {
        let ierr = unsafe { petsc_raw::DMGlobalToLocalBegin(self.dm_p, global.vec_p, mode, local.vec_p) };
        unsafe { chkerrq!(self.world, ierr) }?;
        let ierr = unsafe { petsc_raw::DMGlobalToLocalEnd(self.dm_p, global.vec_p, mode, local.vec_p) };
        unsafe { chkerrq!(self.world, ierr) }
    }

    /// Updates local vectors from global vectors.
    ///
    /// In the [`ADD_VALUES`](InsertMode::ADD_VALUES) case you normally would zero the receiving vector
    /// before beginning this operation. [`INSERT_VALUES`](InsertMode::INSERT_VALUES) is not supported
    /// for DMDA, in that case simply compute the values directly into a global vector instead of a local one.
    pub fn local_to_global(&self, local: &Vector, mode: InsertMode, global: &mut Vector) -> Result<()> {
        let ierr = unsafe { petsc_raw::DMLocalToGlobalBegin(self.dm_p, local.vec_p, mode, global.vec_p) };
        unsafe { chkerrq!(self.world, ierr) }?;
        let ierr = unsafe { petsc_raw::DMLocalToGlobalEnd(self.dm_p, local.vec_p, mode, global.vec_p) };
        unsafe { chkerrq!(self.world, ierr) }
    }

    /// Returns a multi-dimension immutable view that shares data with the underlying vector and is indexed using
    /// the local dimensions.
    ///
    /// # Note
    ///
    /// The C api version of this, (`DMDAVecGetArrayRead`), returns an array using the global dimensions by
    /// applying an offset to the arrays. This method does NOT do that, the view must be indexed starting at zero.
    /// You can get the offsets with [`DM::da_get_corners()`] or [`DM::da_get_ghost_corners()`].
    ///
    /// Also, the underling array has a fortran contiguous layout (column major) whereas the C api swaps the order
    /// of the indexing (so essentially has row major but transposed). This means that in rust you will index the array
    /// normally, but for best performance (i.e., with caching) you should treat it as column major.
    ///
    /// If you view a local vector, than the view will include the ghost corners. Thus the slice returned by
    /// this method could be larger and the offset might be different. In this case [`DM::da_get_ghost_corners()`]
    /// will give you the correct sizes and offsets.
    ///
    /// # Example
    ///
    /// Note, to support complex numbers we use `c(real)` as a shorthand.
    /// Read docs for [`PetscScalar`] for more information.
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # use mpi::traits::*;
    /// # use ndarray::{Dimension, array, s};
    /// # fn c(r: PetscReal) -> PetscScalar { PetscScalar::from(r) }
    /// # fn main() -> petsc_rs::Result<()> {
    /// # let petsc = Petsc::init_no_args()?;
    /// // Note, cargo wont run tests with mpi so this will always be run with
    /// // a single processor, but this example will also work in a multiprocessor
    /// // comm world.
    /// let (m, n) = (5,2);
    /// 
    /// let mut dm = DM::da_create_2d(petsc.world(), DMBoundaryType::DM_BOUNDARY_NONE,
    ///     DMBoundaryType::DM_BOUNDARY_NONE, DMDAStencilType::DMDA_STENCIL_BOX, m, n,
    ///     None, None, 1, 1, None, None)?;
    /// dm.set_from_options()?;
    /// dm.set_up()?;
    ///
    /// let mut global = dm.create_global_vector()?;
    ///
    /// let gs = global.get_global_size()?;
    /// let osr = global.get_ownership_range()?;
    /// let osr_usize = osr.start as usize ..osr.end as usize;
    /// // Note, this follows the global PETSc ordering (not the global natural ordering).
    /// global.assemble_with((0..gs)
    ///         .filter(|i| osr.contains(i))
    ///         .map(|i| (i, c(i as PetscReal))),
    ///     InsertMode::INSERT_VALUES)?;
    /// # let viewer = Viewer::create_ascii_stdout(petsc.world())?;
    /// # global.view_with(Some(&viewer))?;
    ///
    /// // creates immutable 2d view
    /// let g_view = dm.da_vec_view(&global)?;
    /// 
    /// let (xs, ys, _zs, xm, ym, _zm) = dm.da_get_corners()?;
    /// let (gxs, gys, _zs, gxm, gym, _gzm) = dm.da_get_ghost_corners()?;
    ///
    /// // Note because we are viewing a global vector we wont have any ghost corners
    /// assert_eq!(global.get_local_size()?, xm*ym);
    /// if petsc.world().size() > 1 {
    ///     assert_ne!(global.get_local_size()?, gxm*gym);
    /// }
    ///
    /// // Note, standard layout is contiguous C order (row major).
    /// // Also note, a 1d array is always in standard layout.
    /// if g_view.dim().slice()[0] > 1 {
    ///     assert!(!g_view.is_standard_layout());
    /// }
    ///
    /// assert_eq!(g_view.ndim(), 2);
    /// assert_eq!(g_view.dim().slice(), &[gxm as usize, gym as usize]);
    /// // In memory, the 1d slice is ordered.
    /// // Because `g_view` is column major, we `reversed_axes()` to get it to row
    /// // major order. This allows us to use `.as_slice()`. Note, this does not
    /// // change or copy the underling data.
    /// assert_eq!(g_view.view().reversed_axes().as_slice().unwrap(),
    ///     &[c(0.0), c(1.0), c(2.0), c(3.0), c(4.0), c(5.0), c(6.0),
    ///         c(7.0), c(8.0), c(9.0)][osr_usize.clone()]);
    ///
    /// // Ignoring the layout, the array functionally looks like the following.
    /// if petsc.world().size() == 1 {
    ///     let rhs_array = array![[c(0.0), c(5.0)], 
    ///                            [c(1.0), c(6.0)], 
    ///                            [c(2.0), c(7.0)],
    ///                            [c(3.0), c(8.0)],
    ///                            [c(4.0), c(9.0)]];
    ///     assert_eq!(g_view.slice(s![.., ..]).dim(), rhs_array.dim());
    ///     assert_eq!(g_view.slice(s![.., ..]), rhs_array);
    /// } else if petsc.world().size() == 2 {
    ///     let rhs_array = array![[c(0.0), c(3.0)],  // |
    ///                            [c(1.0), c(4.0)],  // | Process 1
    ///                            [c(2.0), c(5.0)],  // |
    ///                            [c(6.0), c(8.0)],  // }
    ///                            [c(7.0), c(9.0)]]; // } Process 2
    ///     assert_eq!(g_view.slice(s![.., ..]).dim(),
    ///         rhs_array.slice(s![xs..(xs+xm), ys..(ys+ym)]).dim());
    ///     assert_eq!(g_view.slice(s![.., ..]), rhs_array.slice(s![xs..(xs+xm), ys..(ys+ym)]));
    /// } else {
    ///     Petsc::set_error(petsc.world(), PetscErrorKind::PETSC_ERR_WRONG_MPI_SIZE, 
    ///         "This example only work with 1 or 2 processors!")?;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// If we wanted to view a local vector we would do something like the following.
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # use mpi::traits::*;
    /// # use ndarray::{Dimension, array, s};
    /// # fn c(r: PetscReal) -> PetscScalar { PetscScalar::from(r) }
    /// # fn main() -> petsc_rs::Result<()> {
    /// # let petsc = Petsc::init_no_args()?;
    /// // Note, cargo wont run tests with mpi so this will always be run with
    /// // a single processor, but this example will also work in two processor
    /// // comm world.
    /// let (m, n) = (5,2);
    /// 
    /// let mut dm = DM::da_create_2d(petsc.world(), DMBoundaryType::DM_BOUNDARY_NONE,
    ///     DMBoundaryType::DM_BOUNDARY_NONE, DMDAStencilType::DMDA_STENCIL_BOX, m, n,
    ///     None, None, 1, 1, None, None)?;
    /// dm.set_from_options()?;
    /// dm.set_up()?;
    ///
    /// let mut global = dm.create_global_vector()?;
    ///
    /// let gs = global.get_global_size()?;
    /// let osr = global.get_ownership_range()?;
    /// // Note, this follows the global PETSc ordering (not the global natural ordering).
    /// global.assemble_with((0..gs)
    ///         .filter(|i| osr.contains(i))
    ///         .map(|i| (i, c(i as PetscReal))),
    ///     InsertMode::INSERT_VALUES)?;
    /// # let viewer = Viewer::create_ascii_stdout(petsc.world())?;
    /// # global.view_with(Some(&viewer))?;
    ///
    /// // We create local vector from global vector
    /// let mut local = dm.get_local_vector()?;
    /// dm.global_to_local(&global, InsertMode::INSERT_VALUES, &mut local)?;
    ///
    /// // creates immutable 2d view
    /// let l_view = dm.da_vec_view(&local)?;
    /// let g_view = dm.da_vec_view(&global)?;
    /// 
    /// let (xs, ys, _zs, xm, ym, _zm) = dm.da_get_corners()?;
    /// let (gxs, gys, _zs, gxm, gym, _gzm) = dm.da_get_ghost_corners()?;
    ///
    /// // Note because we are viewing a local vector we will have the ghost corners
    /// if petsc.world().size() > 1 {
    ///     assert_ne!(local.get_local_size()?, xm*ym);
    ///     assert_eq!(global.get_local_size()?, gxm*gym);
    /// }
    /// assert_eq!(local.get_local_size()?, gxm*gym);
    /// assert_eq!(global.get_local_size()?, xm*ym);
    ///
    /// // Ignoring the layout, the array functionally looks like the following.
    /// if petsc.world().size() == 1 {
    ///     assert_eq!(g_view.slice(s![.., ..]).dim(), l_view.slice(s![.., ..]).dim());
    ///     assert_eq!(g_view.slice(s![.., ..]), l_view.slice(s![.., ..]));
    /// } else if petsc.world().size() == 2 {
    ///     assert_ne!(g_view.slice(s![.., ..]).dim(), l_view.slice(s![.., ..]).dim());
    ///     assert_eq!(g_view.slice(s![.., ..]), l_view.slice(s![xs-gxs..xm, ys-gys..ym]));
    /// } else {
    ///     Petsc::set_error(petsc.world(), PetscErrorKind::PETSC_ERR_WRONG_MPI_SIZE, 
    ///         "This example only work with 1 or 2 processors!")?;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn da_vec_view<'at, 'b>(&self, vec: &'b Vector<'at>) -> Result<crate::vector::VectorView<'at, 'b>> {
        let (xs, ys, zs, xm, ym, zm) = self.da_get_corners()?;
        let (dim, _, _, _, _, _, _, dof, _, _, _, _, _) = self.da_get_info()?;
        let local_size = vec.get_local_size()?;

        let (_gxs, _gys, _gzs, gxm, gym, gzm) = if local_size == xm*ym*zm*dof { 
            (xs, ys, zs, xm, ym, zm)
        } else {
            self.da_get_ghost_corners()?
        };

        if local_size != gxm*gym*gzm*dof {
            seterrq!(self.world, PetscErrorKind::PETSC_ERR_ARG_INCOMP, 
                format!("Vector local size {} is not compatible with DMDA local sizes {} or {}\n",
                    local_size,xm*ym*zm*dof,gxm*gym*gzm*dof))?;
        }

        if dim > 3 || dim < 1 {
            seterrq!(self.world, PetscErrorKind::PETSC_ERR_ARG_CORRUPT, 
                format!("DMDA dimension not 1, 2, or 3, it is {}\n",dim))?;
        }

        let mut array = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::VecGetArrayRead(vec.vec_p, array.as_mut_ptr() as *mut _) };
        unsafe { chkerrq!(vec.world, ierr) }?;

        //let dims = [(gxm*dof) as usize, gym as usize, gzm as usize];
        let dims_r = [gzm as usize, gym as usize, (gxm*dof) as usize];

        let ndarray = unsafe {
            ArrayView::from_shape_ptr(ndarray::IxDyn(&dims_r[(3-dim as usize)..]), array.assume_init())
                .reversed_axes() };

        Ok(crate::vector::VectorView { vec, array: unsafe { array.assume_init() }, ndarray })
    }

    /// Returns a multi-dimension mutable view that shares data with the underlying vector and is indexed using
    /// the local dimensions.
    ///
    /// # Note
    ///
    /// The C api version of this, (`DMDAVecGetArrayRead`), returns an array using the global dimensions by
    /// applying an offset to the arrays. This method does NOT do that, the view must be indexed starting at zero.
    /// You can get the offsets with [`DM::da_get_corners()`] or [`DM::da_get_ghost_corners()`].
    ///
    /// Also, the underling array has a fortran contiguous layout (column major) whereas the C api swaps the order
    /// of the indexing (so essentially has row major but transposed). This means that in rust you will index the array
    /// normally, but for best performance (i.e., with caching) you should treat it as column major.
    ///
    /// If you view a local vector, than the view will include the ghost corners. Thus the slice returned by
    /// this method could be larger and the offset might be different. In this case [`DM::da_get_ghost_corners()`]
    /// will give you the correct sizes and offsets.
    ///
    /// # Example
    ///
    /// Note, to support complex numbers we use `c(real)` as a shorthand.
    /// Read docs for [`PetscScalar`](crate::PetscScalar) for more information.
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # use mpi::traits::*;
    /// # use ndarray::{Dimension, array, s};
    /// # fn c(r: PetscReal) -> PetscScalar { PetscScalar::from(r) }
    /// # fn main() -> petsc_rs::Result<()> {
    /// # let petsc = Petsc::init_no_args()?;
    /// // Note, cargo wont run tests with mpi so this will always be run with
    /// // a single processor, but this example will also work in a multiprocessor
    /// // comm world.
    /// let (m, n) = (5,2);
    /// 
    /// let mut dm = DM::da_create_2d(petsc.world(), DMBoundaryType::DM_BOUNDARY_NONE,
    ///     DMBoundaryType::DM_BOUNDARY_NONE, DMDAStencilType::DMDA_STENCIL_BOX, m, n,
    ///     None, None, 1, 1, None, None)?;
    /// dm.set_from_options()?;
    /// dm.set_up()?;
    ///
    /// let mut global = dm.create_global_vector()?;
    ///
    /// global.set_all(c(0.0))?;
    /// 
    /// let (xs, ys, _zs, xm, ym, _zm) = dm.da_get_corners()?;
    /// let (gxs, gys, _zs, gxm, gym, _gzm) = dm.da_get_ghost_corners()?;
    ///
    /// // Note because we are viewing a global vector we wont have any ghost corners
    /// assert_eq!(global.get_local_size()?, xm*ym);
    /// assert_ne!(global.get_local_size()?, gxm*gxm);
    ///
    /// let mut g_view = dm.da_vec_view_mut(&mut global)?;
    ///
    /// // Note, standard layout is contiguous C order (row major).
    /// // Also note, a 1d array is always in standard layout
    /// if g_view.dim().slice()[0] > 1 {
    ///     assert!(!g_view.is_standard_layout());
    /// }
    ///
    /// // Note, this is automaticly account for the column major layout.
    /// g_view.indexed_iter_mut().map(|(pat, v)| { 
    ///         let s = pat.slice(); 
    ///         ((s[0]+gxs as usize, s[1]+gys as usize), v) 
    ///     })
    ///     .for_each(|((i,j), v)| *v = c((i*2+j) as PetscReal));
    ///
    /// let rhs_array = array![[c(0.0), c(1.0)], 
    ///                        [c(2.0), c(3.0)], 
    ///                        [c(4.0), c(5.0)],
    ///                        [c(6.0), c(7.0)],
    ///                        [c(8.0), c(9.0)]];
    /// assert_eq!(g_view.slice(s![.., ..]).dim(),
    ///     rhs_array.slice(s![gxs..(gxs+gxm), gys..(gys+gym)]).dim());
    /// assert_eq!(g_view.slice(s![.., ..]), rhs_array.slice(s![gxs..(gxs+gxm), gys..(gys+gym)]));
    /// # Ok(())
    /// # }
    /// ```
    pub fn da_vec_view_mut<'at, 'b>(&self, vec: &'b mut Vector<'at>) -> Result<crate::vector::VectorViewMut<'at, 'b>> {
        let (xs, yx, zs, xm, ym, zm) = self.da_get_corners()?;
        let (dim, _, _, _, _, _, _, dof, _, _, _, _, _) = self.da_get_info()?;
        let local_size = vec.get_local_size()?;

        let (_gxs, _gyx, _gzs, gxm, gym, gzm) = if local_size == xm*ym*zm*dof { 
            (xs, yx, zs, xm, ym, zm)
        } else {
            self.da_get_ghost_corners()?
        };

        if local_size != gxm*gym*gzm*dof {
            seterrq!(self.world, PetscErrorKind::PETSC_ERR_ARG_INCOMP, 
                format!("Vector local size {} is not compatible with DMDA local sizes {} or {}\n",
                    local_size,xm*ym*zm*dof,gxm*gym*gzm*dof))?;
        }

        if dim > 3 || dim < 1 {
            seterrq!(self.world, PetscErrorKind::PETSC_ERR_ARG_CORRUPT, 
                format!("DMDA dimension not 1, 2, or 3, it is {}\n",dim))?;
        }

        let mut array = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::VecGetArray(vec.vec_p, array.as_mut_ptr() as *mut _) };
        unsafe { chkerrq!(vec.world, ierr) }?;

        let dims_r = [gzm as usize, gym as usize, (gxm*dof) as usize];

        let ndarray = unsafe {
            ArrayViewMut::from_shape_ptr(ndarray::IxDyn(&dims_r[(3-dim as usize)..]), array.assume_init())
                .reversed_axes() };

        Ok(crate::vector::VectorViewMut { vec, array: unsafe { array.assume_init() }, ndarray })
    }

    /// Sets the names of individual field components in multi-component vectors associated with a DMDA.
    ///
    /// Note, you must call [`DM::set_up()`] before you call this.
    ///
    /// # Parameters
    ///
    /// * `nf` - field number for the DMDA (0, 1, ... dof-1), where dof indicates the number of
    /// degrees of freedom per node within the DMDA.
    /// * `name` - the name of the field (component)
    pub fn da_set_feild_name(&mut self, nf: PetscInt, name: &str) -> crate::Result<()> {
        let name_cs = CString::new(name).expect("`CString::new` failed");
        
        let ierr = unsafe { petsc_raw::DMDASetFieldName(self.dm_p, nf, name_cs.as_ptr()) };
        unsafe { chkerrq!(self.world, ierr) }
    }
    
    /// Gets the ranges of indices in the x, y and z direction that are owned by each process 
    ///
    /// Note: these correspond to the optional final arguments passed to [`DM::da_create_1d()`],
    /// [`DM::da_create_2d()`], and [`DM::da_create_3d()`].
    ///
    /// These numbers are NOT multiplied by the number of dof per node. 
    pub fn da_get_ownership_ranges(&self) -> Result<(Vec<PetscInt>,
        Vec<PetscInt>, Vec<PetscInt>)>
    {
        let mut lx = MaybeUninit::<*const PetscInt>::uninit();
        let mut ly = MaybeUninit::<*const PetscInt>::uninit();
        let mut lz = MaybeUninit::<*const PetscInt>::uninit();

        let dim = self.get_dimension()?;

        if dim > 3 || dim < 1 {
            seterrq!(self.world, PetscErrorKind::PETSC_ERR_ARG_CORRUPT, 
                format!("DMDA dimension not 1, 2, or 3, it is {}\n",dim))?;
        }

        let ierr = unsafe { petsc_raw::DMDAGetOwnershipRanges(self.dm_p, lx.as_mut_ptr(),
            if dim >= 2 { ly.as_mut_ptr() } else { std::ptr::null_mut() }, 
            if dim >= 3 { lz.as_mut_ptr() } else { std::ptr::null_mut() } ) };
        unsafe { chkerrq!(self.world, ierr) }?;

        // SAFETY: Petsc says these are arrays of length comm size
        let lx_vec = {
            let lx_slice = unsafe { std::slice::from_raw_parts(lx.assume_init(), self.world.size() as usize) };
            lx_slice.to_vec()
        };
        let ly_vec = if dim >= 2 {
            let ly_slice = unsafe { std::slice::from_raw_parts(ly.assume_init(), self.world.size() as usize) };
            ly_slice.to_vec()
        } else {
            vec![]
        };
        let lz_vec = if dim >= 2 {
            let lz_slice = unsafe { std::slice::from_raw_parts(lz.assume_init(), self.world.size() as usize) };
            lz_slice.to_vec()
        } else {
            vec![]
        };

        Ok((lx_vec, ly_vec, lz_vec))
    }

    /// Gets the dms in the composite dm created with [`DM::composite_create()`].
    pub fn composite_dms(&self) -> Result<&[Self]> {
        self.composite_dms.as_ref().map(|c| c.as_ref()).ok_or_else(|| 
            seterrq!(self.world, PetscErrorKind::PETSC_ERR_ARG_WRONGSTATE,
                format!("There are no composite dms set, line: {}", line!())).unwrap_err())
    }

    // pub fn composite_dms_mut(&mut self) -> Result<Vec<&mut DM<'a>>> {
    //     if let Some(c) = self.composite_dms.as_mut() {
    //         Ok(c.iter_mut().collect())
    //     } else {
    //         seterrq!(self.world, PetscErrorKind::PETSC_ERR_ARG_WRONGSTATE,
    //             format!("There are no composite dms set, line: {}", line!())).map(|_| unreachable!())
    //     }
    // }

    /// adds a DM vector to a DMComposite 
    pub fn composite_add_dm(&mut self, dm: DM<'a, 'tl>) -> Result<()> {
        let is_dm_comp = self.type_compare(DMType::DMCOMPOSITE)?;
        if is_dm_comp {
            let ierr = unsafe { petsc_raw::DMCompositeAddDM(self.dm_p, dm.dm_p) };
            unsafe { chkerrq!(dm.world, ierr) }?;

            if let Some(c) = self.composite_dms.as_mut() {
                c.push(dm);
            } else {
                self.composite_dms = Some(vec![dm]);
            }

            Ok(())
        } else {
            seterrq!(self.world, PetscErrorKind::PETSC_ERR_ARG_WRONGSTATE,
                    format!("The DM is not a composite DM, line: {}", line!()))
        }
    }

    /// Scatters from a global packed vector into its individual local vectors.
    pub fn composite_scatter<'v, I>(&self, gvec: &'v Vector<'a>, lvecs: I) -> Result<()>
    where
    // TODO: should this be an IntoIter or just a slice? I dont see a case where you wouldn't
    // just have a vec or a slice to the local vectors.
        I: IntoIterator<Item = &'v mut Vector<'a>>,
    {
        if let Some(c) = self.composite_dms.as_ref() {
            let mut lvecs_p =  lvecs.into_iter().map(|v| v.vec_p).collect::<Vec<_>>();

            assert_eq!(lvecs_p.len(), c.len());
            let ierr = unsafe { petsc_raw::DMCompositeScatterArray(self.dm_p, gvec.vec_p, lvecs_p.as_mut_ptr()) };
            unsafe { chkerrq!(self.world, ierr) }
        } else {
            seterrq!(self.world, PetscErrorKind::PETSC_ERR_ARG_WRONGSTATE,
                format!("There are no composite dms set, line: {}", line!()))
        }
    }

    /// Allows one to access the individual packed vectors in their global representation.
    ///
    /// The returned [`Vec`] will have the length of the number of dms in the composite dm.
    pub fn composite_get_access_mut<'bv>(&'bv self, gvec: &'bv mut Vector<'a>) -> Result<Vec<vector::BorrowVectorMut<'a, 'bv>>> {
        // TODO: make a non mut version: look at (https://petsc.org/release/docs/manualpages/Vec/VecLockGet.html#VecLockGet)
        // also look at: https://petsc.org/release/src/dm/impls/composite/pack.c.html#DMCompositeGetAccessArray
        // There is a readonly var that we need to set (if we even can)
        // Or we can just not use DMCompositeGetAccessArray and write our own, because it is just
        // a wrapper around other functions.
        // This internal readonly thing could also cause problems if we return a readonly vec as a mut ref.
        if let Some(c) = self.composite_dms.as_ref() {
            let wanted = (0..c.len() as PetscInt).collect::<Vec<_>>();
            let mut vec_ps = vec![std::ptr::null_mut(); c.len()]; // TODO: can we use MaybeUninit

            let ierr = unsafe { petsc_raw::DMCompositeGetAccessArray(self.dm_p, gvec.vec_p,
                c.len() as PetscInt, wanted.as_ptr(), vec_ps.as_mut_ptr()) };
            unsafe { chkerrq!(self.world, ierr) }?;

            let gvec_rc = Rc::new(&*gvec);
            Ok(vec_ps.into_iter().zip(wanted).map(move |(v_p, i)| {
                let vec = ManuallyDrop::new(Vector { world: self.world, vec_p: v_p });
                let gvec_rc = gvec_rc.clone();
                vector::BorrowVectorMut::new(vec, Some(Box::new(move |borrow_vec| {
                    let i = i;
                    let ierr = unsafe { petsc_raw::DMCompositeRestoreAccessArray(
                        self.dm_p, gvec_rc.vec_p, 1, std::slice::from_ref(&i).as_ptr(),
                        std::slice::from_mut(&mut borrow_vec.vec_p).as_mut_ptr()) };
                    let _ = unsafe { chkerrq!(self.world, ierr) }; // TODO: should I unwrap ?
                })))
            }).collect())

        } else {
            seterrq!(self.world, PetscErrorKind::PETSC_ERR_ARG_WRONGSTATE,
                format!("There are no composite dms set, line: {}", line!())).map(|_| unreachable!())
        }
    }


    /// Gets the index sets for each composed object.
    ///
    /// These could be used to extract a subset of vector entries for a "multi-physics" preconditioner.
    pub fn composite_get_global_indexsets(&self) -> Result<Vec<IS<'a>>> {
        if let Some(c) = self.composite_dms.as_ref() {
            // This Petsc function is a little weird; it allocated the output array
            // and we are expected to free it with `PetscFree`.
            let mut is_array_p = MaybeUninit::uninit();
            let len = c.len();
            let ierr = unsafe { petsc_raw::DMCompositeGetGlobalISs(self.dm_p, is_array_p.as_mut_ptr()) };
            unsafe { chkerrq!(self.world, ierr) }?;

            let is_slice = unsafe { slice::from_raw_parts(is_array_p.assume_init(), len) };

            let ret_vec = is_slice.iter().map(|is_p| IS { world: self.world, is_p: *is_p }).collect();

            let cs_fn_name = CString::new("composite_get_global_indexsets").expect("CString::new failed");
            let cs_file_name = CString::new("dm.rs").expect("CString::new failed");
            
            // Note, `PetscFree` is a macro around `PetscTrFree`
            let ierr = unsafe { (petsc_raw::PetscTrFree.unwrap())(is_array_p.assume_init() as *mut _,
                line!() as i32, cs_fn_name.as_ptr(), cs_file_name.as_ptr()) };
            unsafe { chkerrq!(self.world, ierr) }?;

            Ok(ret_vec)
        } else {
            seterrq!(self.world, PetscErrorKind::PETSC_ERR_ARG_WRONGSTATE,
                format!("There are no composite dms set, line: {}", line!())).map(|_| unreachable!())
        }
    }
    
    /// Gets index sets for each component of a composite local vector.
    ///
    /// # Notes
    ///
    /// At present, a composite local vector does not normally exist. This function is used to provide
    /// index sets for [`Mat::get_local_sub_matrix_mut()`]. In the future, the scatters for each entry
    /// in the DMComposite may be be merged into a single scatter to a composite local vector. The user
    /// should not typically need to know which is being done. 
    ///
    /// To get index sets for pieces of the composite global vector, use [`DM::composite_get_global_indexsets()`]. 
    pub fn composite_get_local_indexsets(&self) -> Result<Vec<IS<'a>>> {
        if let Some(c) = self.composite_dms.as_ref() {
            // This Petsc function is a little weird; it allocated the output array
            // and we are expected to free it with `PetscFree`.
            let mut is_array_p = MaybeUninit::uninit();
            let len = c.len();
            let ierr = unsafe { petsc_raw::DMCompositeGetLocalISs(self.dm_p, is_array_p.as_mut_ptr()) };
            unsafe { chkerrq!(self.world, ierr) }?;

            let is_slice = unsafe { slice::from_raw_parts(is_array_p.assume_init(), len) };

            let ret_vec = is_slice.iter().map(|is_p| IS { world: self.world, is_p: *is_p }).collect();

            let cs_fn_name = CString::new("composite_get_global_indexsets").expect("CString::new failed");
            let cs_file_name = CString::new("dm.rs").expect("CString::new failed");
            
            let ierr = unsafe { (petsc_raw::PetscTrFree.unwrap())(is_array_p.assume_init() as *mut _,
                line!() as i32, cs_fn_name.as_ptr(), cs_file_name.as_ptr()) };
            unsafe { chkerrq!(self.world, ierr) }?;

            Ok(ret_vec)
        } else {
            seterrq!(self.world, PetscErrorKind::PETSC_ERR_ARG_WRONGSTATE,
                format!("There are no composite dms set, line: {}", line!())).map(|_| unreachable!())
        }
    }

    /// This is a WIP, i want to use this instead of doing `if let Some(c) = self.composite_dms.as_ref()`
    /// everywhere.
    fn try_get_composite_dms(&self) -> Result<Option<&Vec<Self>>> {
        let is_dm_comp = self.type_compare(DMType::DMCOMPOSITE)?;
        if is_dm_comp {
            Ok(self.composite_dms.as_ref())
        } else {
            Ok(None)
        }
    }

    /// Add the discretization object for the given DM field 
    ///
    /// Note, The label indicates the support of the field, or is `None` for the entire mesh.
    pub fn add_field(&mut self, label: impl Into<Option<DMLabel<'a>>>, field: impl Into<FieldDisc<'a, 'tl>>) -> Result<()> {
        // TODO: should we make label be an `Rc<DMLabel>`
        // TODO: what type does the dm need to be, if any?
        let field = field.into();
        let is_correct_type = true; // self.type_compare(petsc_raw::DMTYPE_TABLE[DMType::DMPLEX as usize])?;
        if is_correct_type {
            let label: Option<DMLabel> = label.into();
            let ierr = unsafe { petsc_raw::DMAddField(self.dm_p, label.as_ref().map_or(std::ptr::null_mut(),
                |l| l.dml_p), field.as_raw()) };
            unsafe { chkerrq!(self.world, ierr) }?;

            if let Some(f) = self.fields.as_mut() {
                f.push((label, FieldDiscPriv::Known(field)));
            } else {
                self.fields = Some(vec![(label, FieldDiscPriv::Known(field))]);
            }

            Ok(())
        } else {
            seterrq!(self.world, PetscErrorKind::PETSC_ERR_ARG_WRONGSTATE,
                    format!("The DM is not a composite DM, line: {}", line!()))
        }
    }

    /// Sets the inner values of the rust DM struct from C struct based off of the type
    /// 
    /// SAFETY, You should not drop any of the inner values
    ///
    /// If any of the inner values are already set, then they will NOT be dropped
    ///
    /// Right now this is only implemented for DM Composite, for any other type this wont
    /// do anything.
    pub(crate) unsafe fn set_inner_values_for_readonly(dm: &mut Self) -> petsc_raw::PetscErrorCode {
        if Petsc::unwrap_or_abort(dm.type_compare(DMType::DMCOMPOSITE), dm.world()) {
            let len = Petsc::unwrap_or_abort(dm.composite_get_num_dms_petsc(), dm.world());
            let mut dms_p = vec![std::ptr::null_mut(); len as usize]; // use MaybeUninit if we can
            let ierr = petsc_raw::DMCompositeGetEntriesArray(dm.dm_p, dms_p.as_mut_ptr());
            if ierr != 0 { let _ = chkerrq!(dm.world, ierr); return ierr; }

            if let Some(mut old_dms) = dm.composite_dms.take() {
                while !old_dms.is_empty() {
                    let _ = ManuallyDrop::new(old_dms.pop().unwrap());
                }
            }

            dm.composite_dms = Some(dms_p.into_iter().map(|dm_p| {
                let mut this_dm = DM::new(dm.world, dm_p);
                let _ierr = DM::set_inner_values_for_readonly(&mut this_dm);
                //if ierr != 0 { return ierr; }
                this_dm
            }).collect::<Vec<_>>());
            0
        } else if Petsc::unwrap_or_abort(dm.type_compare(DMType::DMDA), dm.world()) {
            0
        } else if Petsc::unwrap_or_abort(dm.type_compare(DMType::DMPLEX), dm.world()) {
            0 // TODO: should we do stuff here
        } else if Petsc::unwrap_or_abort(dm.type_compare(DMType::DMSHELL), dm.world()) {
            // TODO: This is for the case when you dont set a `DM`.
            // If we add DM shell bindings then we will need to
            // change this.
            0
        } else {
            let _ = dbg!(dm.get_type_str());
            todo!()
        }
    }

    /// Gets the [`DM`] method type and name (as a [`String`]). 
    pub fn get_type_str(&self) -> Result<String> {
        let mut s_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::DMGetType(self.dm_p, s_p.as_mut_ptr()) };
        unsafe { chkerrq!(self.world, ierr) }?;
        
        let c_str: &CStr = unsafe { CStr::from_ptr(s_p.assume_init()) };
        let str_slice: &str = c_str.to_str().unwrap();
        Ok(str_slice.to_owned())
    }

    /// Create a label of the given name if it does not already exist 
    pub fn create_label(&mut self, name: &str) -> Result<()> {
        let name_cs = CString::new(name).expect("`CString::new` failed");
        let ierr = unsafe { petsc_raw::DMCreateLabel(self.dm_p, name_cs.as_ptr()) };
        unsafe { chkerrq!(self.world, ierr) }
    }

    /// Return the label of a given name if present
    ///
    /// Note: Some of the default labels in a DMPlex will be:
    /// * "depth"       - Holds the depth (co-dimension) of each mesh point
    /// * "celltype"    - Holds the topological type of each cell
    /// * "ghost"       - If the DM is distributed with overlap, this marks the cells and faces in the overlap
    /// * "Cell Sets"   - Mirrors the cell sets defined by GMsh and ExodusII
    /// * "Face Sets"   - Mirrors the face sets defined by GMsh and ExodusII
    /// * "Vertex Sets" - Mirrors the vertex sets defined by GMsh
    pub fn get_label(&self, name: &str) -> Result<Option<DMLabel<'a>>> {
        let name_cs = CString::new(name).expect("`CString::new` failed");
        let mut dm_label = MaybeUninit::<*mut petsc_raw::_p_DMLabel>::uninit();
        let ierr = unsafe { petsc_raw::DMGetLabel(self.dm_p, name_cs.as_ptr(), dm_label.as_mut_ptr()) };
        unsafe { chkerrq!(self.world, ierr) }?;

        let dm_label = NonNull::new(unsafe { dm_label.assume_init() } );

        let mut label = dm_label.map(|nn_dml_p| DMLabel { world: self.world, dml_p: nn_dml_p.as_ptr() });
        if let Some(l) = label.as_mut() {
            unsafe { l.reference()? };
        }
        Ok(label)
    }

    /// Complete labels that are being used for FEM BC
    #[cfg(petsc_version_3_16_dev)]
    unsafe fn complete_boundary_label_internal(&self, ds: &DS, field: PetscInt, bd_num: PetscInt, label: &mut DMLabel) -> Result<()> {
        // In the C code, this is a static function declared/defined in `src/dm/interface/dm.c`
        // So this function is just a port of it.
        let mut duplicate = false;
        if let Some(ref fields) = self.fields {
            let (_, disc) = &fields[field as usize];
            let id = match disc {
                FieldDiscPriv::Known(fd) => fd.get_class_id(),
                FieldDiscPriv::Unknown(po) => po.get_class_id(),
            }?;
            if id == petsc_raw::PETSCFE_CLASSID {
                let nbd = ds.get_num_boundary()?;
                for bd in 0..PetscInt::min(nbd, bd_num) {
                    let (_, _, _, l, _, _, _) = ds.get_boundary_info(bd)?;
                    duplicate = label.as_raw() == l.as_raw();
                    if duplicate { break; }
                }
                if !duplicate {
                    let mut plex_p = MaybeUninit::uninit();
                    let ierr = petsc_raw::DMConvert(self.dm_p,
                        petsc_raw::DMPLEX.as_ptr() as *const _, plex_p.as_mut_ptr());
                    chkerrq!(self.world(), ierr)?;
                    let plex = ManuallyDrop::new(DM::new(self.world(), plex_p.assume_init()));
                    plex.plex_label_complete(label)?;
                }
            }
        }

        Ok(())
    }

    /// Complete labels that are being used for FEM BC
    #[cfg(petsc_version_3_15)]
    unsafe fn complete_boundary_label_internal(&self, ds: &DS, field: PetscInt, bd_num: PetscInt, labelname: &str) -> Result<()> {
        // In the C code, this is a static function declared/defined in `src/dm/interface/dm.c`
        // So this function is just a port of it.
        let mut duplicate = false;
        if let Some(ref fields) = self.fields {
            let (_, disc) = &fields[field as usize];
            let id = match disc {
                FieldDiscPriv::Known(fd) => fd.get_class_id(),
                FieldDiscPriv::Unknown(po) => po.get_class_id(),
            }?;
            if let Some(mut label) = self.get_label(labelname)? {
                if id == petsc_raw::PETSCFE_CLASSID {
                    let nbd = ds.get_num_boundary()?;
                    for bd in 0..PetscInt::min(nbd, bd_num) {
                        let (_, _, ln, _, _, _) = ds.get_boundary_info(bd)?;
                        duplicate = labelname == ln;
                        if duplicate { break; }
                    }
                    if !duplicate {
                        let mut plex_p = MaybeUninit::uninit();
                        let ierr = petsc_raw::DMConvert(self.dm_p,
                            petsc_raw::DMPLEX.as_ptr() as *const _, plex_p.as_mut_ptr());
                        chkerrq!(self.world(), ierr)?;
                        let plex = ManuallyDrop::new(DM::new(self.world(), plex_p.assume_init()));
                        plex.plex_label_complete(&mut label)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Add an essential boundary condition to the model. (WIP Wrapper function)
    ///
    /// In the C API you would call `DMAddBoundary` with the type being `DM_BC_ESSENTIAL`.
    ///
    /// This API is for PETSc `v3.16-dev.0`
    ///
    /// # Parameters
    ///
    /// * `self` - The DM, with a PetscDS that matches the problem being constrained
    /// * `name` - The BC name
    /// * `label` - The label defining constrained points
    /// * `values` - An array of values for constrained points
    /// * `field` - The field to constrain
    /// * `comps` - An array of constrained component numbers
    /// * `bc_user_func` - A pointwise function giving boundary values
    ///
    /// ## `bc_user_func` Parameters
    ///
    /// * `dim` - the spatial dimension
    /// * `time` - the time of the current point
    /// * `x` - coordinates of the current point
    /// * `nc` - the number of field components
    /// * `bcval` *(output)* - output values at the current point
    ///
    /// # Example
    ///
    /// Look at [`snes-ex12`](https://gitlab.com/petsc/petsc-rs/-/blob/main/examples/snes/src/ex12.rs)
    #[cfg(any(petsc_version_3_16_dev, doc))]
    pub fn add_boundary_essential<F1>(&mut self, name: &str, label: &mut DMLabel, values: &[PetscInt],
        field: PetscInt, comps: &[PetscInt], bc_user_func: F1) -> Result<PetscInt>
    where
        F1: Fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
    {
        // TODO: PETSc c func does a lot of extra stuff with C macros:
        // PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
        // PetscValidLogicalCollectiveEnum(dm, type, 2);
        // PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 4);
        // PetscValidLogicalCollectiveInt(dm, Nv, 5);
        // PetscValidLogicalCollectiveInt(dm, field, 7);
        // PetscValidLogicalCollectiveInt(dm, Nc, 8);
        // ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
        // ierr = DMCompleteBoundaryLabel_Internal(dm, ds, field, PETSC_MAX_INT, label);CHKERRQ(ierr);
        // ierr = PetscDSAddBoundary(ds, type, name, label, Nv, values, field, Nc, comps, bcFunc, bcFunc_t, ctx, bd);CHKERRQ(ierr);
        // return 0;

        let _ = self.get_ds_or_create()?;
        let ds = self.try_get_ds().unwrap();
        unsafe { self.complete_boundary_label_internal(ds, field, PetscInt::MAX, label)?; }
        self.get_ds_or_create()?.add_boundary_essential(name, label, values, field, comps, bc_user_func)
    }

    /// Add an essential boundary condition to the model. (WIP Wrapper function)
    ///
    /// In the C API you would call `DMAddBoundary` with the type being `DM_BC_ESSENTIAL`.
    ///
    /// This API is for PETSc `v3.16-dev.0`
    ///
    /// # Parameters
    ///
    /// * `self` - The DM, with a PetscDS that matches the problem being constrained
    /// * `name` - The BC name
    /// * `label` - The label defining constrained points
    /// * `values` - An array of values for constrained points
    /// * `field` - The field to constrain
    /// * `comps` - An array of constrained component numbers
    /// * `bc_user_func` - A pointwise function giving boundary values
    /// * `bc_user_func_t` - A pointwise function giving the time deriative of the boundary values
    ///
    /// ## `bc_user_func`/`bc_user_func_t` Parameters
    ///
    /// * `dim` - the spatial dimension
    /// * `time` - the time of the current point
    /// * `x` - coordinates of the current point
    /// * `nc` - the number of field components
    /// * `bcval` *(output)* - output values at the current point
    #[cfg(any(petsc_version_3_16_dev, doc))]
    pub fn add_boundary_essential_with_dt<F1, F2>(&mut self, name: &str, label: &mut DMLabel, values: &[PetscInt],
        field: PetscInt, comps: &[PetscInt], bc_user_func: F1, bc_user_func_t: F2) -> Result<PetscInt>
    where
        F1: Fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
        F2: Fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
    {
        let _ = self.get_ds_or_create()?;
        let ds = self.try_get_ds().unwrap();
        unsafe { self.complete_boundary_label_internal(ds, field, PetscInt::MAX, label)?; }
        self.get_ds_or_create()?.add_boundary_essential_with_dt(name, label, values, field, comps, bc_user_func, bc_user_func_t)
    }

    /// Add an essential boundary condition to the model. (WIP Wrapper function)
    ///
    /// In the C API you would call `DMAddBoundary` with the type being `DM_BC_ESSENTIAL`.
    ///
    /// This API is for PETSc `v3.15`
    ///
    /// # Parameters
    ///
    /// * `self` - The DM, with a PetscDS that matches the problem being constrained
    /// * `name` - The BC name
    /// * `labelname` - The label defining constrained points
    /// * `field` - The field to constrain
    /// * `comps` - An array of constrained component numbers
    /// * `ids` - An array of ids for constrained points
    /// * `bc_user_func` - A pointwise function giving boundary values
    ///
    /// ## `bc_user_func` Parameters
    ///
    /// * `dim` - the spatial dimension
    /// * `time` - the time of the current point
    /// * `x` - coordinates of the current point
    /// * `nc` - the number of field components
    /// * `bcval` *(output)* - output values at the current point
    #[cfg(any(petsc_version_3_15, doc))]
    pub fn add_boundary_essential<F1>(&mut self, name: &str, labelname: &str, field: PetscInt,
        comps: &[PetscInt], ids: &[PetscInt], bc_user_func: F1) -> Result<()>
    where
        F1: Fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
    {
        let _ = self.get_ds_or_create()?;
        let ds = self.try_get_ds().unwrap();
        unsafe { self.complete_boundary_label_internal(ds, field, PetscInt::MAX, labelname)?; }
        self.get_ds_or_create()?.add_boundary_essential(name, labelname, field, comps, ids, bc_user_func)
    }

    /// Add an essential boundary condition to the model. (WIP Wrapper function)
    ///
    /// In the C API you would call `DMAddBoundary` with the type being `DM_BC_ESSENTIAL`.
    ///
    /// This API is for PETSc `v3.15`
    ///
    /// # Parameters
    ///
    /// * `self` - The DM, with a PetscDS that matches the problem being constrained
    /// * `name` - The BC name
    /// * `labelname` - The label defining constrained points
    /// * `field` - The field to constrain
    /// * `comps` - An array of constrained component numbers
    /// * `ids` - An array of ids for constrained points
    /// * `bc_user_func` - A pointwise function giving boundary values
    /// * `bc_user_func_t` - A pointwise function giving the time deriative of the boundary values
    ///
    /// ## `bc_user_func`/`bc_user_func_t` Parameters
    ///
    /// * `dim` - the spatial dimension
    /// * `time` - the time of the current point
    /// * `x` - coordinates of the current point
    /// * `nc` - the number of field components
    /// * `bcval` *(output)* - output values at the current point
    #[cfg(any(petsc_version_3_15, doc))]
    pub fn add_boundary_essential_with_dt<F1, F2>(&mut self, name: &str, labelname: &str, field: PetscInt,
        comps: &[PetscInt], ids: &[PetscInt], bc_user_func: F1, bc_user_func_t: F2) -> Result<()>
    where
        F1: Fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
        F2: Fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
    {
        let _ = self.get_ds_or_create()?;
        let ds = self.try_get_ds().unwrap();
        unsafe { self.complete_boundary_label_internal(ds, field, PetscInt::MAX, labelname)?; }
        self.get_ds_or_create()?.add_boundary_essential_with_dt(name, labelname, field, comps, ids, bc_user_func, bc_user_func_t)
    }

    /// Add a natural boundary condition to the model. (WIP Wrapper function)
    ///
    /// In the C API you would call `DMAddBoundary` with the type being `DM_BC_NATURAL`.
    ///
    /// This API is for PETSc `v3.16-dev.0`
    ///
    /// # Parameters
    ///
    /// * `self` - The DM, with a PetscDS that matches the problem being constrained
    /// * `name` - The BC name
    /// * `label` - The label defining constrained points
    /// * `values` - An array of values for constrained points
    /// * `field` - The field to constrain
    /// * `comps` - An array of constrained component numbers
    /// * `bc_user_func` - A pointwise function giving boundary values
    ///
    /// # Example
    ///
    /// Look at [`snes-ex12`](https://gitlab.com/petsc/petsc-rs/-/blob/main/examples/snes/src/ex12.rs)
    #[cfg(any(petsc_version_3_16_dev, doc))]
    pub fn add_boundary_natural<F1>(&mut self, name: &str, label: &mut DMLabel, values: &[PetscInt],
        field: PetscInt, comps: &[PetscInt], bc_user_func: F1) -> Result<PetscInt>
    where
        F1: Fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
    {
        let _ = self.get_ds_or_create()?;
        let ds = self.try_get_ds().unwrap();
        unsafe { self.complete_boundary_label_internal(ds, field, PetscInt::MAX, label)?; }
        self.get_ds_or_create()?.add_boundary_natural(name, label, values, field, comps, bc_user_func)
    }

    /// Add a natural boundary condition to the model. (WIP Wrapper function)
    ///
    /// In the C API you would call `DMAddBoundary` with the type being `DM_BC_NATURAL`.
    ///
    /// This API is for PETSc `v3.16-dev.0`
    ///
    /// # Parameters
    ///
    /// * `self` - The DM, with a PetscDS that matches the problem being constrained
    /// * `name` - The BC name
    /// * `label` - The label defining constrained points
    /// * `values` - An array of values for constrained points
    /// * `field` - The field to constrain
    /// * `comps` - An array of constrained component numbers
    /// * `bc_user_func` - A pointwise function giving boundary values
    /// * `bc_user_func_t` - A pointwise function giving the time deriative of the boundary values
    #[cfg(any(petsc_version_3_16_dev, doc))]
    pub fn add_boundary_natural_with_dt<F1, F2>(&mut self, name: &str, label: &mut DMLabel, values: &[PetscInt],
        field: PetscInt, comps: &[PetscInt], bc_user_func: F1, bc_user_func_t: F2) -> Result<PetscInt>
    where
        F1: Fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
        F2: Fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
    {
        let _ = self.get_ds_or_create()?;
        let ds = self.try_get_ds().unwrap();
        unsafe { self.complete_boundary_label_internal(ds, field, PetscInt::MAX, label)?; }
        self.get_ds_or_create()?.add_boundary_natural_with_dt(name, label, values, field, comps, bc_user_func, bc_user_func_t)
    }

    /// Add an essential boundary condition to the model. (WIP Wrapper function)
    ///
    /// In the C API you would call `DMAddBoundary` with the type being `DM_BC_NATURAL`.
    ///
    /// This API is for PETSc `v3.15`
    ///
    /// # Parameters
    ///
    /// * `self` - The DM, with a PetscDS that matches the problem being constrained
    /// * `name` - The BC name
    /// * `labelname` - The label defining constrained points
    /// * `field` - The field to constrain
    /// * `comps` - An array of constrained component numbers
    /// * `ids` - An array of ids for constrained points
    #[cfg(any(petsc_version_3_15, doc))]
    pub fn add_boundary_natural<F1>(&mut self, name: &str, labelname: &str, field: PetscInt,
        comps: &[PetscInt], ids: &[PetscInt], bc_user_func: F1) -> Result<()>
    where
        F1: Fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
    {
        let _ = self.get_ds_or_create()?;
        let ds = self.try_get_ds().unwrap();
        unsafe { self.complete_boundary_label_internal(ds, field, PetscInt::MAX, labelname)?; }
        self.get_ds_or_create()?.add_boundary_natural(name, labelname, field, comps, ids, bc_user_func)
    }

    /// Add an essential boundary condition to the model. (WIP Wrapper function)
    ///
    /// In the C API you would call `DMAddBoundary` with the type being `DM_BC_ESSENTIAL`.
    ///
    /// This API is for PETSc `v3.15`
    ///
    /// # Parameters
    ///
    /// * `self` - The DM, with a PetscDS that matches the problem being constrained
    /// * `name` - The BC name
    /// * `labelname` - The label defining constrained points
    /// * `field` - The field to constrain
    /// * `comps` - An array of constrained component numbers
    /// * `ids` - An array of ids for constrained points
    /// * `bc_user_func` - A pointwise function giving boundary values
    /// * `bc_user_func_t` - A pointwise function giving the time deriative of the boundary values
    #[cfg(any(petsc_version_3_15, doc))]
    pub fn add_boundary_natural_with_dt<F1, F2>(&mut self, name: &str, labelname: &str, field: PetscInt,
        comps: &[PetscInt], ids: &[PetscInt], bc_user_func: F1, bc_user_func_t: F2) -> Result<()>
    where
        F1: Fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
        F2: Fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
    {
        let _ = self.get_ds_or_create()?;
        let ds = self.try_get_ds().unwrap();
        unsafe { self.complete_boundary_label_internal(ds, field, PetscInt::MAX, labelname)?; }
        self.get_ds_or_create()?.add_boundary_natural_with_dt(name, labelname, field, comps, ids, bc_user_func, bc_user_func_t)
    }

    // TODO: should these be unsafe functions, or is having the callback be unsafe enough
    /// Add a essential or natural field boundary condition to the model. (WIP Wrapper function)
    ///
    /// In the C API you would call `DMAddBoundary` with the `bctype` being `DM_BC_*_FIELD`.
    ///
    /// Note, the functions `bc_user_func` and `bc_user_func` can not be closures. And they
    /// Will take pointers instead of slices. 
    ///
    /// This API is for PETSc `v3.16-dev.0`
    ///
    /// # Parameters
    ///
    /// * `self` - The DM, with a PetscDS that matches the problem being constrained
    /// * `bctype` - The type of condition, e.g. DM_BC_ESSENTIAL_FIELD (Dirichlet),
    /// or DM_BC_NATURAL_FIELD (Neumann). Must end in field.
    /// * `name` - The BC name
    /// * `label` - The label defining constrained points
    /// * `values` - An array of values for constrained points
    /// * `field` - The field to constrain
    /// * `comps` - An array of constrained component numbers
    /// * `bc_user_func` - A pointwise function giving boundary values
    /// * `bc_user_func_t` - A pointwise function giving the time deriative of the boundary values, or `None`
    ///
    /// ## `bc_user_func`/`bc_user_func_t` Parameters
    ///
    /// * `dim` - the spatial dimension
    /// * `nf` - the number of fields
    /// * `u_off` - the offset into u[] and u_t[] for each field
    /// * `u_off_x` - the offset into u_x[] for each field
    /// * `u` - each field evaluated at the current point
    /// * `u_t` - the time derivative of each field evaluated at the current point
    /// * `u_x` - the gradient of each field evaluated at the current point
    /// * `a_off` - the offset into a[] and a_t[] for each auxiliary field
    /// * `a_off_x` - the offset into a_x[] for each auxiliary field
    /// * `a` - each auxiliary field evaluated at the current point
    /// * `a_t` - the time derivative of each auxiliary field evaluated at the current point
    /// * `a_x` - the gradient of auxiliary each field evaluated at the current point
    /// * `t` - current time
    /// * `x` - coordinates of the current point
    /// * `nc` - number of constant parameters
    /// * `consts` - constant parameters
    /// * `bcval` *(output)* - output values at the current point
    #[cfg(any(petsc_version_3_16_dev, doc))]
    pub fn add_boundary_field_raw(&mut self, bctype: DMBoundaryConditionType, name: &str, label: &mut DMLabel, values: &[PetscInt],
        field: PetscInt, comps: &[PetscInt],
        bc_user_func: impl Into<Option<unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar)>>,
        bc_user_func_t: impl Into<Option<unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar)>>) -> Result<PetscInt>
    {
        let _ = self.get_ds_or_create()?;
        let ds = self.try_get_ds().unwrap();
        unsafe { self.complete_boundary_label_internal(ds, field, PetscInt::MAX, label)?; }
        self.get_ds_or_create()?.add_boundary_field_raw(bctype, name, label, values, field, comps, bc_user_func, bc_user_func_t)
    }

    /// Add a essential or natural field boundary condition to the model. (WIP Wrapper function)
    ///
    /// In the C API you would call `DMAddBoundary` with the `bctype` being `DM_BC_*_FIELD`.
    ///
    /// This API is for PETSc `v3.15`
    ///
    /// # Parameters
    ///
    /// * `self` - The DM, with a PetscDS that matches the problem being constrained
    /// * `bctype` - The type of condition, e.g. DM_BC_ESSENTIAL_FIELD (Dirichlet),
    /// or DM_BC_NATURAL_FIELD (Neumann). Must end in field.
    /// * `name` - The BC name
    /// * `labelname` - The label defining constrained points
    /// * `field` - The field to constrain
    /// * `comps` - An array of constrained component numbers
    /// * `ids` - An array of ids for constrained points
    /// * `bc_user_func` - A pointwise function giving boundary values
    /// * `bc_user_func_t` - A pointwise function giving the time deriative of the boundary values, or `None`
    ///
    /// ## `bc_user_func`/`bc_user_func_t` Parameters
    ///
    /// * `dim` - the spatial dimension
    /// * `nf` - the number of fields
    /// * `u_off` - the offset into u[] and u_t[] for each field
    /// * `u_off_x` - the offset into u_x[] for each field
    /// * `u` - each field evaluated at the current point
    /// * `u_t` - the time derivative of each field evaluated at the current point
    /// * `u_x` - the gradient of each field evaluated at the current point
    /// * `a_off` - the offset into a[] and a_t[] for each auxiliary field
    /// * `a_off_x` - the offset into a_x[] for each auxiliary field
    /// * `a` - each auxiliary field evaluated at the current point
    /// * `a_t` - the time derivative of each auxiliary field evaluated at the current point
    /// * `a_x` - the gradient of auxiliary each field evaluated at the current point
    /// * `t` - current time
    /// * `x` - coordinates of the current point
    /// * `nc` - number of constant parameters
    /// * `consts` - constant parameters
    /// * `bcval` *(output)* - output values at the current point
    #[cfg(any(petsc_version_3_15, doc))]
    pub fn add_boundary_field_raw(&mut self, bctype: DMBoundaryConditionType, name: &str, labelname: &str, field: PetscInt,
        comps: &[PetscInt], ids: &[PetscInt],
        bc_user_func: impl Into<Option<unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar)>>,
        bc_user_func_t: impl Into<Option<unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar)>>) -> Result<()>
    {
        let _ = self.get_ds_or_create()?;
        let ds = self.try_get_ds().unwrap();
        unsafe { self.complete_boundary_label_internal(ds, field, PetscInt::MAX, labelname)?; }
        self.get_ds_or_create()?.add_boundary_field_raw(bctype, name, labelname, field, comps, ids, bc_user_func, bc_user_func_t)
    }
    
    /// Add a essential field boundary condition to the model. (WIP Wrapper function)
    ///
    /// In the C API you would call `DMAddBoundary` with the `bctype` being `DM_BC_ESSENTIAL_BD_FIELD`.
    ///
    /// Note, the functions `bc_user_func` and `bc_user_func` can not be closures. And they
    /// Will take pointers instead of slices. 
    ///
    /// This API is for PETSc `v3.16-dev.0`
    ///
    /// # Parameters
    ///
    /// * `self` - The DM, with a PetscDS that matches the problem being constrained
    /// * `name` - The BC name
    /// * `label` - The label defining constrained points
    /// * `values` - An array of values for constrained points
    /// * `field` - The field to constrain
    /// * `comps` - An array of constrained component numbers
    /// * `bc_user_func` - A pointwise function giving boundary values
    /// * `bc_user_func_t` - A pointwise function giving the time deriative of the boundary values, or `None`
    ///
    /// ## `bc_user_func`/`bc_user_func_t` Parameters
    ///
    /// * `dim` - the spatial dimension
    /// * `nf` - the number of fields
    /// * `u_off` - the offset into u[] and u_t[] for each field
    /// * `u_off_x` - the offset into u_x[] for each field
    /// * `u` - each field evaluated at the current point
    /// * `u_t` - the time derivative of each field evaluated at the current point
    /// * `u_x` - the gradient of each field evaluated at the current point
    /// * `a_off` - the offset into a[] and a_t[] for each auxiliary field
    /// * `a_off_x` - the offset into a_x[] for each auxiliary field
    /// * `a` - each auxiliary field evaluated at the current point
    /// * `a_t` - the time derivative of each auxiliary field evaluated at the current point
    /// * `a_x` - the gradient of auxiliary each field evaluated at the current point
    /// * `t` - current time
    /// * `x` - coordinates of the current point
    /// * `n` - facet normal at the current point
    /// * `nc` - number of constant parameters
    /// * `consts` - constant parameters
    /// * `bcval` *(output)* - output values at the current point
    #[cfg(any(petsc_version_3_16_dev, doc))]
    pub fn add_boundary_essential_bd_field_raw(&mut self, name: &str, label: &mut DMLabel, values: &[PetscInt],
        field: PetscInt, comps: &[PetscInt],
        bc_user_func: impl Into<Option<unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            PetscReal, *const PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar)>>,
        bc_user_func_t: impl Into<Option<unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            PetscReal, *const PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar)>>) -> Result<PetscInt>
    {
        let _ = self.get_ds_or_create()?;
        let ds = self.try_get_ds().unwrap();
        unsafe { self.complete_boundary_label_internal(ds, field, PetscInt::MAX, label)?; }
        self.get_ds_or_create()?.add_boundary_essential_bd_field_raw(name, label, values, field, comps, bc_user_func, bc_user_func_t)
    }

    /// Add a essential field boundary condition to the model. (WIP Wrapper function)
    ///
    /// In the C API you would call `DMAddBoundary` with the `bctype` being `DM_BC_ESSENTIAL_BD_FIELD`.
    ///
    /// Note, the functions `bc_user_func` and `bc_user_func` can not be closures. And they
    /// Will take pointers instead of slices. 
    ///
    /// This API is for PETSc `v3.15`
    ///
    /// # Parameters
    ///
    /// * `self` - The DM, with a PetscDS that matches the problem being constrained
    /// * `name` - The BC name
    /// * `label` - The label defining constrained points
    /// * `values` - An array of values for constrained points
    /// * `field` - The field to constrain
    /// * `comps` - An array of constrained component numbers
    /// * `bc_user_func` - A pointwise function giving boundary values
    /// * `bc_user_func_t` - A pointwise function giving the time deriative of the boundary values, or `None`
    ///
    /// ## `bc_user_func`/`bc_user_func_t` Parameters
    ///
    /// * `dim` - the spatial dimension
    /// * `nf` - the number of fields
    /// * `u_off` - the offset into u[] and u_t[] for each field
    /// * `u_off_x` - the offset into u_x[] for each field
    /// * `u` - each field evaluated at the current point
    /// * `u_t` - the time derivative of each field evaluated at the current point
    /// * `u_x` - the gradient of each field evaluated at the current point
    /// * `a_off` - the offset into a[] and a_t[] for each auxiliary field
    /// * `a_off_x` - the offset into a_x[] for each auxiliary field
    /// * `a` - each auxiliary field evaluated at the current point
    /// * `a_t` - the time derivative of each auxiliary field evaluated at the current point
    /// * `a_x` - the gradient of auxiliary each field evaluated at the current point
    /// * `t` - current time
    /// * `x` - coordinates of the current point
    /// * `n` - facet normal at the current point
    /// * `nc` - number of constant parameters
    /// * `consts` - constant parameters
    /// * `bcval` *(output)* - output values at the current point
    #[cfg(any(petsc_version_3_15, doc))]
    pub fn add_boundary_essential_bd_field_raw(&mut self, name: &str, labelname: &str, field: PetscInt,
        comps: &[PetscInt], ids: &[PetscInt],
        bc_user_func: impl Into<Option<unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            PetscReal, *const PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar)>>,
        bc_user_func_t: impl Into<Option<unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            PetscReal, *const PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar)>>) -> Result<()>
    {
        let _ = self.get_ds_or_create()?;
        let ds = self.try_get_ds().unwrap();
        unsafe { self.complete_boundary_label_internal(ds, field, PetscInt::MAX, labelname)?; }
        self.get_ds_or_create()?.add_boundary_essential_bd_field_raw(name, labelname, field, comps, ids, bc_user_func, bc_user_func_t)
    }

    /// Create the discrete systems for the DM based upon the fields added to the DM
    ///
    /// Note, If the label has a DS defined, it will be replaced. Otherwise, it will be added to the DM.
    pub fn create_ds(&mut self) -> Result<()> {
        let ierr = unsafe { petsc_raw::DMCreateDS(self.dm_p) };
        unsafe { chkerrq!(self.world, ierr) }?;

        let mut ds_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::DMGetDS(self.dm_p, ds_p.as_mut_ptr()) };
        unsafe { chkerrq!(self.world, ierr) }?;

        self.ds = Some(DS::new(self.world, unsafe { ds_p.assume_init() }));
        unsafe { self.ds.as_mut().unwrap().reference()?; }

        Ok(())
    }

    /// Remove all discrete systems from the DM.
    pub fn clear_ds(&mut self) -> Result<()> {
        let ierr = unsafe { petsc_raw::DMClearDS(self.dm_p) };
        unsafe { chkerrq!(self.world, ierr) }?;

        let _ = self.ds.take();
        
        Ok(())
    }

    /// Returns an [`Option`] to a reference to the [DS](DS).
    ///
    /// For this to return `Some`, you must call [`DM::create_ds()`] first.
    ///
    /// Note, this does not return a [`Result`](crate::Result) because it can never
    /// fail, instead it will return `None`.
    pub fn try_get_ds(&self) -> Option<&DS<'a, 'tl>> {
        self.ds.as_ref()
    }

    /// Returns an [`Option`] to a mutable reference to the [DS](DS).
    ///
    /// For this to return `Some`, you must call [`DM::create_ds()`] first.
    ///
    /// Note, this does not return a [`Result`](crate::Result) because it can never
    /// fail, instead it will return `None`.
    pub fn try_get_ds_mut(&mut self) -> Option<&mut DS<'a, 'tl>> {
        self.ds.as_mut()
    }

    /// Gets the [DS](DS), or creates the default [`DS`].
    pub fn get_ds_or_create(&mut self) -> Result<&mut DS<'a, 'tl>> {
        let mut ds_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::DMGetDS(self.dm_p, ds_p.as_mut_ptr()) };
        unsafe { chkerrq!(self.world, ierr) }?;

        self.ds = Some(DS::new(self.world, unsafe { ds_p.assume_init() }));
        unsafe { self.ds.as_mut().unwrap().reference()?; }

        Ok(self.ds.as_mut().unwrap())
    }

    /// Get the description of mesh periodicity
    ///
    /// # Outputs (in order)
    ///
    /// If `Some` then the DM is periodic:
    /// * `max_cell` - Over distances greater than this, we can assume a point has crossed over
    /// to another sheet, when trying to localize cell coordinates.
    /// * `L` - If we assume the mesh is a torus, this is the length of each coordinate
    /// * `bd` - This describes the type of periodicity in each topological dimension
    pub fn get_periodicity(&self) -> Result<Option<(&[PetscReal], &[PetscReal], &[DMBoundaryType])>> {
        let dim = self.get_dimension()? as usize;
        let mut per = ::std::mem::MaybeUninit::uninit();
        let mut max_cell = ::std::mem::MaybeUninit::uninit();
        let mut l = ::std::mem::MaybeUninit::uninit();
        let mut db = ::std::mem::MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::DMGetPeriodicity(self.dm_p, per.as_mut_ptr(),
            max_cell.as_mut_ptr(), l.as_mut_ptr(), db.as_mut_ptr()) };
        unsafe { chkerrq!(self.world, ierr) }?;

        // TODO: is this correct?
        if unsafe { per.assume_init() }.into() {
            Ok(unsafe { Some((
                slice::from_raw_parts(max_cell.assume_init(), dim),
                slice::from_raw_parts(l.assume_init(), dim),
                slice::from_raw_parts(db.assume_init(), dim))) } )
        } else {
            Ok(None)
        }
    }

    // TODO: maybe make the `dyn FnMut(...)` into a typedef so that the caller can use it more easily
    /// This projects the given function into the function space provided, putting the coefficients in a local vector.
    ///
    /// # Parameters
    ///
    /// * `time` - The time
    /// * `mode` - The insertion mode for values
    /// * `local` - Local vector to write into
    /// * `user_f` - The coordinate functions to evaluate, one per field
    ///     * `dim` - The spatial dimension
    ///     * `t` - Current time
    ///     * `x` - Coordinates of the current point
    ///     * `nf` - The number of field components
    ///     * `u` *(output)* - The output field values
    ///
    /// # Example
    ///
    /// Note, to support complex numbers we use `c(real)` as a shorthand.
    /// Read docs for [`PetscScalar`](crate::PetscScalar) for more information.
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # use mpi::traits::*;
    /// # use std::slice;
    /// # use ndarray::{Dimension, array, s};
    /// # fn c(r: PetscReal) -> PetscScalar { PetscScalar::from(r) }
    /// # #[cfg(feature = "petsc-use-complex-unsafe")]
    /// # fn slice_abs_diff_eq(s1: &[PetscScalar], s2: &[PetscScalar], tol: PetscReal) -> bool {
    /// # s1.len() == s2.len() && s1.iter().zip(s2).all(|(a,b)| (a-b).norm() < tol) }
    /// # #[cfg(not(feature = "petsc-use-complex-unsafe"))]
    /// # fn slice_abs_diff_eq(s1: &[PetscScalar], s2: &[PetscScalar], tol: PetscReal) -> bool {
    /// # s1.len() == s2.len() && s1.iter().zip(s2).all(|(a,b)| (a-b).abs() < tol) }
    /// # fn main() -> petsc_rs::Result<()> {
    /// # let petsc = Petsc::init_no_args()?;
    /// // Note: this example will only work in a uniprocessor comm world.
    /// let simplex = true;
    /// let mut dm = DM::plex_create_box_mesh(petsc.world(), 2, simplex, (2,2,0), None, (6.0,6.0,0.0), None, true)?;
    /// dm.set_name("Mesh")?;
    /// dm.set_from_options()?;
    /// # dm.view_with(None)?;
    /// // The Simplicial cells created by `plex_create_box_mesh` have the following
    /// // layout, where the `*`s are the points that will be used in the closure
    /// // given to `project_function_local`.
    /// // 6-┌ ─ ─ ─ ─ ─ ┬ ─ ─ ─ ─ ─ ┐
    /// //   | \         | \         |
    /// //   |   \   *   |   \   *   |
    /// //   |     \     |     \     |
    /// //   |   *   \   |   *   \   |
    /// //   |         \ |         \ |
    /// // 3-├ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ┤
    /// //   | \         | \         |
    /// // 2-|   \   *   |   \   *   |
    /// //   |     \     |     \     |
    /// // 1-|   *   \   |   *   \   |
    /// //   |         \ |         \ |
    /// // 0-└ ─ ─ ─ ─ ─ ┴ ─ ─ ─ ─ ─ ┘
    /// //   |   |   |   |           |
    /// //   0   1   2   3           6
    ///
    /// // Most of this is just boilerplate
    /// let dim = dm.get_dimension()?;
    /// let mut fe1 = FEDisc::create_default(dm.world(), dim, 1, simplex, None, None)?;
    /// let mut fe2 = FEDisc::create_default(dm.world(), dim, 1, simplex, None, None)?;
    /// dm.add_field(None, fe1)?;
    /// dm.add_field(None, fe2)?;
    /// # dm.view_with(None)?;
    ///
    /// dm.create_ds()?;
    /// let _ = dm.get_coordinate_dm_or_create()?;
    /// # dm.view_with(None)?;
    /// # dm.get_coordinate_dm_or_create()?.view_with(None)?;
    ///
    /// let mut local = dm.create_local_vector()?;
    ///
    /// // Rust has trouble knowing we want Box<dyn _> (and not Box<_>) without the explicate type signature.
    /// // Thus we need to define `funcs` outside of the `project_function_local` function call.
    /// let funcs: [Box<dyn FnMut(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> petsc_rs::Result<()>>; 2]
    ///     = [Box::new(|dim, time, x, nc, u| {
    ///         u[0] = c(x[0]*x[0] + x[1]*x[1]);
    ///         # println!("fe1: dim: {}, time: {}, x: {:?}, nc: {}, u: {:?}", dim, time, x, nc, u);
    ///         Ok(())
    ///     }),
    ///     Box::new(|dim, time, x, nc, u| {
    ///         u[0] = c(x[0]*x[0] - x[1]*x[1]);
    ///         # println!("fe2: dim: {}, time: {}, x: {:?}, nc: {}, u: {:?}", dim, time, x, nc, u);
    ///         Ok(())
    ///     })];
    /// dm.project_function_local(0.0, InsertMode::INSERT_ALL_VALUES, &mut local, funcs)?;
    /// # petsc_println_sync!(petsc.world(), "[Process {}] {:.5}", petsc.world().rank(), *local.view()?)?;
    ///
    /// # // TODO: give explanation on whys the points are used and when
    /// assert!(slice_abs_diff_eq(local.view()?.as_slice().unwrap(),
    ///     &[c(2.0), c(0.0), c(17.0), c(-15.0), c(8.0), c(0.0), c(29.0), c(21.0),
    ///         c(17.0), c(15.0), c(50.0), c(0.0), c(32.0), c(0.0), c(29.0), c(-21.0)], 10e-15));
    /// # Ok(())
    /// # }
    /// ```
    // TODO: should this take mut self? i dont think it needs to. The closure we give it is called
    // during this function call so it can be `FnMut` while self doesn't need to be.
    pub fn project_function_local<'sl>(&self, time: PetscReal, mode: InsertMode, local: &mut Vector,
        user_funcs: impl IntoIterator<Item = Box<dyn FnMut(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'sl>>)
        -> Result<()>
    {
        let nf = self.get_num_fields()? as usize;
        if let Some(fields) = self.fields.as_ref() {
            if fields.len() != nf {
                // This should never happen if user uses the rust api to add fields
                seterrq!(self.world, PetscErrorKind::PETSC_ERR_COR,
                    "Number of fields in C strutc and rust struct do not match.")?;
            }
        } else {
            seterrq!(self.world, PetscErrorKind::PETSC_ERR_ARG_WRONGSTATE,
                "No fields have been set.")?;
        }

        let trampoline_datas = user_funcs.into_iter().map(|closure_anchor| Box::pin(DMProjectFunctionTrampolineData { 
            user_f: closure_anchor })).collect::<Vec<_>>();

        if trampoline_datas.len() != nf {
            seterrq!(self.world, PetscErrorKind::PETSC_ERR_USER_INPUT,
                format!("Expected {} functions in `user_funcs`, but there were {}.", nf, trampoline_datas.len()))?;
        }

        unsafe extern "C" fn dm_project_function_local_trampoline(dim: PetscInt, time: PetscReal, x: *const PetscReal,
            nf: PetscInt, u: *mut petsc_raw::PetscScalar, ctx: *mut ::std::os::raw::c_void) -> petsc_raw::PetscErrorCode
        {
            let trampoline_data: Pin<&mut DMProjectFunctionTrampolineData> = std::mem::transmute(ctx);

            let x_slice = slice::from_raw_parts(x, dim as usize);
            let u_slice = slice::from_raw_parts_mut(u as *mut _, nf as usize);
            
            (trampoline_data.get_mut().user_f)(dim, time, x_slice, nf, u_slice)
                .map_or_else(|err| err.kind as i32, |_| 0)
        }

        // When it comes to function pointers casting in NOT your friend. It causes the pointers 
        // to be curropted. You MUST be VERY explicit with the types. All in all, the following
        // will get the function pointers in options correctly (I think). Once we have them, we can 
        // then manipulate them how we see fit.
        let mut trampoline_funcs: Vec<::std::option::Option<
            unsafe extern "C" fn(arg1: PetscInt, arg2: PetscReal, arg3: *const PetscReal, arg4: PetscInt,
                arg5: *mut petsc_raw::PetscScalar, arg6: *mut ::std::os::raw::c_void,) -> petsc_raw::PetscErrorCode, >>
            = vec![Some(dm_project_function_local_trampoline); nf];
        let mut trampoline_data_refs = trampoline_datas.iter().map(|td| unsafe { std::mem::transmute(td.as_ref()) }).collect::<Vec<_>>();

        let ierr = unsafe { petsc_raw::DMProjectFunctionLocal(
            self.dm_p, time, trampoline_funcs.as_mut_ptr(),
            trampoline_data_refs.as_mut_ptr(),
            mode, local.vec_p) };
        unsafe { chkerrq!(self.world, ierr) }?;

        Ok(())
    }

    /// This projects the given function into the function space provided, putting the coefficients in a vector.
    ///
    /// Under the hood uses [`DM::project_function_local()`] and [`DM::local_to_global()`].
    pub fn project_function<'sl>(&self, time: PetscReal, mode: InsertMode, global: &mut Vector<'a>,
        user_funcs: impl IntoIterator<Item = Box<dyn FnMut(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'sl>>)
        -> Result<()>
    {
        let mut local = self.get_local_vector()?;
        self.project_function_local(time, mode, &mut local, user_funcs)?;
        self.local_to_global(&local, mode, global)
    }

    // TODO: should these be unsafe
    /// This projects the given function of the input fields into the function space provided,
    /// putting the coefficients in a local vector.
    pub fn project_field_local_raw<'vl>(&mut self, time: PetscReal, localu: impl Into<Option<&'vl Vector<'a>>>, mode: InsertMode, localx: &mut Vector,
        user_funcs: impl IntoIterator<Item = Option<unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar)>>)
        -> Result<()>
    where
        'a: 'vl,
    {
        let nf = self.get_num_fields()? as usize;
        if let Some(fields) = self.fields.as_ref() {
            if fields.len() != nf {
                // This should never happen if user uses the rust api to add fields
                seterrq!(self.world, PetscErrorKind::PETSC_ERR_COR,
                    "Number of fields in C strutc and rust struct do not match.")?;
            }
        } else {
            seterrq!(self.world, PetscErrorKind::PETSC_ERR_ARG_WRONGSTATE,
                "No fields have been set.")?;
        }

        let mut fn_ptrs = user_funcs.into_iter().collect::<Vec<Option<unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar)>>>();

        if fn_ptrs.len() != nf {
            seterrq!(self.world, PetscErrorKind::PETSC_ERR_USER_INPUT,
                format!("Expected {} functions in `user_funcs`, but there were {}.", nf, fn_ptrs.len()))?;
        }

        let mut localu_owned;
        let localu = if let Some(localu) = localu.into() {
            localu
        } else {
            localu_owned = Vector { vec_p: localx.vec_p, world: localx.world };
            unsafe { localu_owned.reference()?; }
            &localu_owned
        };

        let ierr = unsafe { petsc_raw::DMProjectFieldLocal(
            self.dm_p, time, localu.vec_p, fn_ptrs.as_mut_ptr() as *mut _,
            mode, localx.vec_p) };
        unsafe { chkerrq!(self.world, ierr) }?;

        Ok(())
    }

    /// This projects the given function of the input fields into the function space provided,
    /// putting the coefficients in a global vector.
    pub fn project_field_raw<'vl>(&mut self, time: PetscReal, u: impl Into<Option<&'vl Vector<'a>>>, mode: InsertMode, x: &mut Vector,
        user_funcs: impl IntoIterator<Item = Option<unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar)>>)
        -> Result<()>
    where
        'a: 'vl,
    {
        let nf = self.get_num_fields()? as usize;
        if let Some(fields) = self.fields.as_ref() {
            if fields.len() != nf {
                // This should never happen if user uses the rust api to add fields
                seterrq!(self.world, PetscErrorKind::PETSC_ERR_COR,
                    "Number of fields in C strutc and rust struct do not match.")?;
            }
        } else {
            seterrq!(self.world, PetscErrorKind::PETSC_ERR_ARG_WRONGSTATE,
                "No fields have been set.")?;
        }

        let mut fn_ptrs = user_funcs.into_iter().collect::<Vec<Option<unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar)>>>();

        if fn_ptrs.len() != nf {
            seterrq!(self.world, PetscErrorKind::PETSC_ERR_USER_INPUT,
                format!("Expected {} functions in `user_funcs`, but there were {}.", nf, fn_ptrs.len()))?;
        }

        let mut u_owned;
        let u = if let Some(u) = u.into() {
            u
        } else {
            u_owned = Vector { vec_p: x.vec_p, world: x.world };
            unsafe { u_owned.reference()?; }
            &u_owned
        };

        let ierr = unsafe { petsc_raw::DMProjectField(
            self.dm_p, time, u.vec_p, fn_ptrs.as_mut_ptr() as *mut _,
            mode, x.vec_p) };
        unsafe { chkerrq!(self.world, ierr) }?;

        Ok(())
    }

    /// This function computes the L_2 difference between the gradient of a function u and
    /// an FEM interpolant solution grad `u_h`.
    ///
    /// # Parameters 
    ///
    /// * `time` - the time
    /// * `u_h` - The coefficient vector, a global vector
    /// * `user_funcs` - The functions to evaluate for each field component
    /// * *output* - The diff ||u - u_h||_2
    // TODO: should this take mut self? i dont think it needs to. The closure we give it is called
    // during this function call so it can be `FnMut` while self doesn't need to be.
    pub fn compute_l2_diff<'sl>(&self, time: PetscReal, u_h: &Vector,
        user_funcs: impl IntoIterator<Item = Box<dyn FnMut(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'sl>>)
        -> Result<PetscReal>
    {
        let mut diff = MaybeUninit::uninit();

        let nf = self.get_num_fields()? as usize;
        if let Some(fields) = self.fields.as_ref() {
            if fields.len() != nf {
                // This should never happen if user uses the rust api to add fields
                seterrq!(self.world, PetscErrorKind::PETSC_ERR_COR,
                    "Number of fields in C strutc and rust struct do not match.")?;
            }
        } else {
            seterrq!(self.world, PetscErrorKind::PETSC_ERR_ARG_WRONGSTATE,
                "No fields have been set.")?;
        }

        let trampoline_datas = user_funcs.into_iter().map(|closure_anchor| Box::pin(DMProjectFunctionTrampolineData { 
            user_f: closure_anchor })).collect::<Vec<_>>();

        if trampoline_datas.len() != nf {
            seterrq!(self.world, PetscErrorKind::PETSC_ERR_USER_INPUT,
                format!("Expected {} functions in `user_funcs`, but there were {}.", nf, trampoline_datas.len()))?;
        }

        unsafe extern "C" fn dm_compute_ls_diff_trampoline(dim: PetscInt, time: PetscReal, x: *const PetscReal,
            nf: PetscInt, u: *mut petsc_raw::PetscScalar, ctx: *mut ::std::os::raw::c_void) -> petsc_raw::PetscErrorCode
        {
            let trampoline_data: Pin<&mut DMProjectFunctionTrampolineData> = std::mem::transmute(ctx);

            let x_slice = slice::from_raw_parts(x, dim as usize);
            let u_slice = slice::from_raw_parts_mut(u as *mut _, nf as usize);
            
            (trampoline_data.get_mut().user_f)(dim, time, x_slice, nf, u_slice)
                .map_or_else(|err| err.kind as i32, |_| 0)
        }

        // When it comes to function pointers casting in NOT your friend. It causes the pointers 
        // to be curropted. You MUST be VERY explicit with the types. All in all, the following
        // will get the function pointers in options correctly (I think). Once we have them, we can 
        // then manipulate them how we see fit.
        let mut trampoline_funcs: Vec<::std::option::Option<
            unsafe extern "C" fn(arg1: PetscInt, arg2: PetscReal, arg3: *const PetscReal, arg4: PetscInt,
                arg5: *mut petsc_raw::PetscScalar, arg6: *mut ::std::os::raw::c_void,) -> petsc_raw::PetscErrorCode, >>
            = vec![Some(dm_compute_ls_diff_trampoline); nf];
        let mut trampoline_data_refs = trampoline_datas.iter().map(|td| unsafe { std::mem::transmute(td.as_ref()) }).collect::<Vec<_>>();

        let ierr = unsafe { petsc_raw::DMComputeL2Diff(
            self.dm_p, time, trampoline_funcs.as_mut_ptr(),
            trampoline_data_refs.as_mut_ptr(),
            u_h.vec_p, diff.as_mut_ptr()) };
        unsafe { chkerrq!(self.world, ierr) }?;

        Ok(unsafe { diff.assume_init() })
    }

    /// Form the integral over the specified boundary from the global input X using pointwise
    /// functions specified by the user.
    ///
    /// # Parameters
    ///
    /// * `x`       - Global input [`Vector`]
    /// * `label`   - The boundary [`DMLabel`]
    /// * `vals`    - The label values to use, or `None` for all values
    /// * `func`    - The function to integrate along the boundary
    pub fn plex_compute_bd_integral_raw<'vll, 'll, 'lal: 'll>(&mut self, x: &Vector, label: impl Into<Option<&'ll DMLabel<'lal>>>,
        vals: impl Into<Option<&'vll [PetscInt]>>, func: unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            PetscReal, *const PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar)) -> Result<PetscScalar>
    {
        let mut integral = MaybeUninit::uninit();

        let fn_ptr: Option<unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            PetscReal, *const PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar)> = Some(func);
        let vals: Option<&[PetscInt]> = vals.into();

        let ierr = unsafe { petsc_raw::DMPlexComputeBdIntegral(
            self.dm_p, x.vec_p, label.into().as_raw(), 
            vals.as_ref().map_or(petsc_raw::PETSC_DETERMINE, |&vl| vl.len() as PetscInt),
            vals.as_ref().map_or(std::ptr::null(), |&vl| vl.as_ptr()),
            std::mem::transmute(fn_ptr), integral.as_mut_ptr(), std::ptr::null_mut()) };
        unsafe { chkerrq!(self.world, ierr) }?;

        Ok(unsafe { integral.assume_init() }.into())
    }

    /// Gets the DM that prescribes coordinate layout and scatters between global and local coordinates.
    pub fn get_coordinate_dm_or_create<'sl>(&'sl mut self) -> Result<Rc<DM<'a, 'tl>>> {
        if self.coord_dm.is_some() {
            Ok(self.coord_dm.as_ref().unwrap().clone())
        } else {
            let mut dm2_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::DMGetCoordinateDM(self.dm_p, dm2_p.as_mut_ptr()) };
            unsafe { chkerrq!(self.world, ierr) }?;

            let mut coord_dm = DM::new(self.world, unsafe { dm2_p.assume_init() });
            unsafe { coord_dm.reference()?; }
            self.coord_dm = Some(Rc::new(coord_dm));

            Ok(self.coord_dm.as_ref().unwrap().clone())
        }
    }

    /// Gets the DM that prescribes coordinate layout and scatters between global and local coordinates.
    ///
    /// If the coordinate hasn't been set, then you must call
    /// [`DM::get_coordinate_dm_or_create()`] or [`DM::set_coordinate_dm()`]
    /// for this to return a `Some`.
    pub fn try_get_coordinate_dm<'sl>(&'sl self) -> Option<Rc<DM<'a, 'tl>>> {
        self.coord_dm.clone()
    }

    /// Sets the DM that prescribes coordinate layout and scatters between global and local coordinates 
    pub fn set_coordinate_dm(&mut self, coord_dm: Rc<DM<'a, 'tl>>) -> Result<()> {
        let ierr = unsafe { petsc_raw::DMSetCoordinateDM(self.dm_p, coord_dm.dm_p) };
        unsafe { chkerrq!(self.world, ierr) }?;
        self.coord_dm = Some(coord_dm);
        Ok(())
    }

    /// Determine whether the mesh has a label of a given name 
    pub fn has_label(&self, labelname: &str) -> Result<bool> {
        let labelname_cs = CString::new(labelname).expect("`CString::new` failed");
        let mut res = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::DMHasLabel(self.dm_p, labelname_cs.as_ptr(), res.as_mut_ptr()) };
        unsafe { chkerrq!(self.world, ierr) }?;
        Ok(unsafe { res.assume_init().into() })
    }

    /// Mark all faces on the boundary 
    ///
    /// If `val` is `None` then that marker values will be some value in the closure (or 1 if none are found).
    pub fn plex_mark_boundary_faces(&mut self, val: impl Into<Option<PetscInt>>, label: &mut DMLabel) -> Result<()> {
        let ierr = unsafe { petsc_raw::DMPlexMarkBoundaryFaces(
            self.dm_p,val.into().unwrap_or(petsc_raw::PETSC_DETERMINE), label.dml_p) };
        unsafe { chkerrq!(self.world, ierr) }
    }

    /// Copy the fields and discrete systems for one [`DM`] into this [`DM`]
    pub fn copy_disc_from(&mut self, other: &DM) -> Result<()> {
        let ierr = unsafe { petsc_raw::DMCopyDisc(other.dm_p, self.dm_p) };
        unsafe { chkerrq!(self.world, ierr) }?;

        let nf = self.get_num_fields()?;
        let mut new_fields = vec![];
        for i in 0..nf {
            let res = self.get_field_from_c_struct(i)?;
            new_fields.push(res);
        }
        self.fields = Some(new_fields);

        Ok(())
    }

    /// Internal function
    fn get_field_from_c_struct(&self, f: PetscInt) -> Result<(Option<DMLabel<'a>>, FieldDiscPriv<'a, 'tl>)>
    { 
        let mut dml_p = MaybeUninit::uninit();
        let mut f_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::DMGetField(self.dm_p, f, dml_p.as_mut_ptr(), f_p.as_mut_ptr() as *mut _) };
        unsafe { chkerrq!(self.world, ierr) }?;

        let dm_label = NonNull::new(unsafe { dml_p.assume_init() } );
        let mut label = dm_label.map(|nn_dml_p| DMLabel { world: self.world, dml_p: nn_dml_p.as_ptr() });
        if let Some(l) = label.as_mut() {
            unsafe { l.reference()?; }
        }

        let mut field = crate::PetscObjectStruct { world: self.world, po_p: unsafe { f_p.assume_init() } };
        unsafe { field.reference()?; }
        Ok((label, FieldDiscPriv::Unknown(field)))
    }

    /// Get the coarse mesh from which this was obtained by refinement.
    pub fn get_coarse_dm_mut(&mut self) -> Result<Option<&mut DM<'a, 'tl>>> {
        // There is no non-mut version of this fruntion because if the coarse dm exists only in
        // the C struct, then we have to retrieve it which requiers a mutable refrence to self.
        // So having a non-mut version that still requiers a mut ref to self would be pointless.
        // Having a try_get version would be confusing because this function doesn't create the 
        // coarse dm if it doesn't exist, it just retrieves it from the C API if it's there.
        if self.coarse_dm.is_some() {
            Ok(Some(self.coarse_dm.as_mut().unwrap().as_mut()))
        } else {
            let mut dm2_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::DMGetCoarseDM(self.dm_p, dm2_p.as_mut_ptr()) };
            unsafe { chkerrq!(self.world, ierr) }?;

            let dm_nn_p = NonNull::new(unsafe { dm2_p.assume_init() } );
            self.coarse_dm = dm_nn_p.map(|dm_nn_p| Box::new(DM::new(self.world, dm_nn_p.as_ptr())));
            if let Some(cdm) = self.coarse_dm.as_mut() {
                unsafe { cdm.as_mut().reference()?; }
            }

            Ok(self.coarse_dm.as_mut().map(|box_dm| box_dm.as_mut()))
        }
    }

    // TODO: a lot of work still needs to be done on the dm cloning

    /// Clones and return a [`BorrowDM`] that depends on the lifetime of self.
    ///
    /// You should only call this method if you are cloneing a [`DM`] that is a `DMPLEX`.
    ///
    /// The new [`BorrowDM`] will give you access to the DMPLEX data including closures
    /// owned by the original [`DM`].
    pub fn clone_shallow(&self) -> Result<BorrowDM<'a, 'tl, '_>> {
        // The goal here is to make sure that the the dm we are cloning from lives longer that the new_dm
        // This insures that the closures are valid. We can do this by using `BorrowDM`.
        // TODO: We should use some form of interior mutability to allow the use to still
        // edit the original DM aslong as they dont drop it or change existing closures.
        // IDK if this is possible.
    
        let new_dm = unsafe { self.clone_unchecked()? };
    
        Ok(BorrowDM { owned_dm: new_dm, _phantom: PhantomData })
    }

    /// Unsafe clone
    ///
    /// This is `unsafe` because for some DM implementations this is a shallow clone,
    /// the result of which may share (referent counted) information with its parent.
    /// For example, clone applied to a DMPLEX object will result in a new DMPLEX that
    /// shares the topology with the original DMPLEX. It does not share the PetscSection
    /// of the original DM.
    ///
    /// You can also call [`DM::clone`] which is panic is cases where DM:clone is invalid.
    /// Or you can call [`DM::clone_shallow`] which will tie the new [`DM`]s lifetime to that
    /// of the original.
    ///
    /// Note, the rust trampoline data is stored in a `RefCell`. 
    // If we refrence count the trampoline data, including the closures and have everything
    // in it be immutable, is this unsafe?
    pub unsafe fn clone_unchecked(&self) -> Result<DM<'a, 'tl>> {
        Ok(if self.type_compare(DMType::DMCOMPOSITE)? {
            let c = self.try_get_composite_dms()?.unwrap();
            DM::composite_create(self.world, c.iter().cloned())?
        } else if self.type_compare(DMType::DMPLEX)? {
            // Note, DMPlex just refrence counts the underlying data and shollow copies.
            let mut dm2_p = MaybeUninit::uninit();
            let ierr = petsc_raw::DMClone(self.dm_p, dm2_p.as_mut_ptr());
            chkerrq!(self.world, ierr)?;
            let mut dm = DM::new(self.world, dm2_p.assume_init());
            let nf = dm.get_num_fields()?;
            let mut new_fields = vec![];
            for i in 0..nf {
                let res = dm.get_field_from_c_struct(i)?;
                new_fields.push(res);
            }
            dm.fields = Some(new_fields);
            dm
        } else {
            let mut dm2_p = MaybeUninit::uninit();
            let ierr = petsc_raw::DMClone(self.dm_p, dm2_p.as_mut_ptr());
            chkerrq!(self.world, ierr)?;
            DM::new(self.world, dm2_p.assume_init())
        })
    }
    
    /// Will clone the DM.
    ///
    /// This method is the exact same as [`DM::clone()`] but will return a [`Result`].
    ///
    /// Note, you can NOT clone a `DMPLEX`.
    /// Instead use [`DM::clone_shallow()`] or [`DM::clone_unchecked()`].
    pub fn clone_result(&self) -> Result<DM<'a, 'tl>> {
        // TODO: the docs say this is a shallow clone. How should we deal with this for rust
        // (rust (and the caller) thinks/expects it is a deep clone)
        // TODO: Also what should we do for DM composite type, i get an error when DMClone calls
        // `DMGetDimension` and then `DMSetDimension` (the dim is -1).

        // TODO: we don't need to do this. If the closure is defined with a static lifetime then
        // we should be fine
        if self.type_compare(DMType::DMPLEX)? {
            seterrq!(self.world, PetscErrorKind::PETSC_ERR_ARG_WRONGSTATE,
                "You can not clone a DMPLEX with `DM::clone()`, use `DM::clone_shallow()` or `DM::clone_unchecked()` instead.")?;
        }

        unsafe { self.clone_unchecked() }
    }

    /// Iterates over self and recursive calls to [`DM::get_coarse_dm_mut`] until we get a `None`
    /// from [`DM::get_coarse_dm_mut()`], running `f` on the dm each time.
    ///
    /// After running `f`, it copies the discrete system from `self` to the coarse dm using [`DM::copy_disc_from()`].
    pub fn plex_for_each_coarse_dm<F>(&mut self, mut f: F) -> Result<()>
    where
        F: FnMut(&mut DM) -> Result<()>
    {
        let dm = ManuallyDrop::new(DM::new(self.world, self.dm_p));
        let mut cdm = self;
        loop {
            f(cdm)?;
            if cdm.dm_p != dm.dm_p {
                cdm.copy_disc_from(&dm)?;
            }
            if let Some(this_cdm) = cdm.get_coarse_dm_mut()? {
                cdm = this_cdm;
            } else {
                break;
            }
        }
        Ok(())
    }

    /// Get the auxiliary vector for region specified by the given label and value (indicating the region).
    ///
    /// Note: If no auxiliary vector is found for this (label, value), `(label: None, value: 0)` is checked
    /// as well.
    ///
    /// Only avalable for PETSc `v3.16-dev.0`
    // TODO: is it safe to return an Rc
    #[cfg(any(petsc_version_3_16_dev, doc))]
    pub fn get_auxiliary_vec<'ll, 'lal: 'll>(&self, label: impl Into<Option<&'ll DMLabel<'lal>>>, value: PetscInt) -> Result<Option<Rc<Vector<'a>>>> {
        let mut vec_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::DMGetAuxiliaryVec(self.dm_p, label.into().as_raw(), value,
            vec_p.as_mut_ptr()) };
        unsafe { chkerrq!(self.world, ierr) }?;

        let vec_p_nn = NonNull::new(unsafe { vec_p.assume_init() } );
        let mut vec = vec_p_nn.map(|nn_vec_p| Vector { world: self.world, vec_p: nn_vec_p.as_ptr() });
        if let Some(v) = vec.as_mut() {
            unsafe { v.reference()? };
        }

        Ok(vec.map(|aux| Rc::new(aux)))
    }

    /// Determines whether a PETSc [`DM`] is of a particular type.
    pub fn type_compare(&self, type_kind: DMType) -> Result<bool> {
        self.type_compare_str(&type_kind.to_string())
    }

    /// Get the integer ids in a label
    pub fn get_label_id_is(&self, name: &str) -> Result<Option<IS<'a>>> {
        let name_cs = CString::new(name).expect("`CString::new` failed");
        let mut is_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::DMGetLabelIdIS(self.dm_p, name_cs.as_ptr(), is_p.as_mut_ptr()) };
        unsafe { chkerrq!(self.world, ierr) }?;

        let nn_is_p = NonNull::new(unsafe { is_p.assume_init() } );
        Ok(nn_is_p.map(|nn_is_p| IS { world: self.world, is_p: nn_is_p.as_ptr() }))
    }

    /// Sets the prefix used for searching for all options of [`DM`] in the database.
    ///
    /// Same as [PetscObject::set_options_prefix()], but also sets prefix for some internal types.
    pub fn dm_set_options_prefix(&mut self, prefix: &str) -> crate::Result<()> {
        let name_cs = ::std::ffi::CString::new(prefix).expect("`CString::new` failed");
        
        let ierr = unsafe { crate::petsc_raw::DMSetOptionsPrefix(self.as_raw(), name_cs.as_ptr()) };
        unsafe { chkerrq!(self.world(), ierr) }
    }

    /// Set [`SNES`] residual evaluation function.
    ///
    /// Note, [`SNES::set_function()`] is normally used, but it calls this function internally
    /// because the user context is actually associated with the DM. This makes the interface
    /// consistent regardless of whether the user interacts with a DM or not. If DM took a more
    /// central role at some later date, this could become the primary method of setting the residual.
    ///
    /// # Parameters
    ///
    /// * `user_f` - A closure used to convey the nonlinear function to be solved by SNES
    ///     * `snes` - the snes context
    ///     * `x` - state at which to evaluate residual
    ///     * `f` *(output)* - vector to put residual (function value)
    pub fn snes_set_function<F>(&mut self, user_f: F) -> Result<()>
    where
        F: FnMut(&SNES<'a, '_, '_>, &Vector<'a>, &mut Vector<'a>) -> std::result::Result<(), DomainOrPetscError> + 'tl,
    {
        // TODO: look at how rsmpi did the trampoline stuff:
        // https://github.com/rsmpi/rsmpi/blob/82e1d357/src/collective.rs#L1684
        // They used libffi, that could be a safer way to do it.

        let closure_anchor = Box::new(user_f);

        let trampoline_data = Box::pin(SNESFunctionTrampolineData { 
            world: self.world, user_f: closure_anchor });

        // drop old trampoline_data
        let _ = self.snes_function_trampoline_data.take();

        // I think this function is called by the SNES, not the dm
        unsafe extern "C" fn snes_function_trampoline(snes_p: *mut petsc_raw::_p_SNES, x_p: *mut petsc_raw::_p_Vec,
            f_p: *mut petsc_raw::_p_Vec, ctx: *mut std::os::raw::c_void) -> petsc_raw::PetscErrorCode
        {
            // SAFETY: We construct ctx to be a Pin<Box<SNESFunctionTrampolineData>> but pass it in as a *void
            // Box<T> is equivalent to *T (or &T) for ffi. Because the SNES owns the closure we can make sure
            // everything in it (and the closure its self) lives for at least as long as this function can be
            // called.
            // We don't construct a Box<> because we dont want to drop anything
            let trampoline_data: Pin<&mut SNESFunctionTrampolineData> = std::mem::transmute(ctx);

            // We don't want to drop anything, we are just using this to turn pointers 
            // of the underlining types (i.e. *mut petsc_raw::_p_SNES) into references
            // of the rust wrapper types.
            // Note, SNES has optional members that might have to be dropped, but because
            // we only give immutable access to the user_f we don't have to worry about that
            // as they will all stay `None`.
            // If `Vector` ever has optional parameters, they MUST be dropped manually.
            // SAFETY: even though snes is mut and thus we can set optional parameters, we don't
            // as we dont expose the mut to the user closure, we only use it with `set_jacobian_domain_error`
            let mut snes = ManuallyDrop::new(SNES::new(trampoline_data.world, snes_p));

            let mut dm_p = MaybeUninit::uninit();
            let ierr = petsc_raw::SNESGetDM(snes_p, dm_p.as_mut_ptr());
            if ierr != 0 { let _ = chkerrq!(trampoline_data.world, ierr); return ierr; }
            let mut dm = DM::new(trampoline_data.world, dm_p.assume_init());
            let ierr = DM::set_inner_values_for_readonly(&mut dm);
            if ierr != 0 { return ierr; }
            snes.dm = Some(dm); // Note, because snes is not dropped, snes.dm wont be either

            let x = ManuallyDrop::new(Vector { world: trampoline_data.world, vec_p: x_p });
            let mut f = ManuallyDrop::new(Vector { world: trampoline_data.world, vec_p: f_p });
            
            (trampoline_data.get_mut().user_f)(&snes, &x, &mut f)
                .map_or_else(|err| match err {
                    DomainOrPetscError::DomainErr => {
                        let perr = snes.set_function_domain_error();
                        match perr {
                            Ok(_) => 0,
                            Err(perr) => perr.kind as i32
                        }
                    },
                    DomainOrPetscError::PetscErr(perr) => perr.kind as i32
                }, |_| 0)
        }

        let ierr = unsafe { petsc_raw::DMSNESSetFunction(
            self.dm_p, Some(snes_function_trampoline), 
            std::mem::transmute(trampoline_data.as_ref())) }; // this will also erase the lifetimes
        unsafe { chkerrq!(self.world, ierr) }?;
        
        self.snes_function_trampoline_data = Some(trampoline_data);

        Ok(())
    }

    /// Set [`SNES`] Jacobian evaluation function, with the jacobian and the preconditioner being the same.
    ///
    /// Note, [`SNES::set_jacobian_single_mat()`] is normally used, but it calls this function internally
    /// because the user context is actually associated with the DM. This makes the interface consistent
    /// regardless of whether the user interacts with a DM or not. If DM took a more central role at some
    /// later date, this could become the primary method of setting the Jacobian. 
    ///
    /// # Parameters
    ///
    /// * `user_f` - A closure used to convey the Jacobian evaluation routine.
    ///     * `snes` - the snes context
    ///     * `x` - input vector, the Jacobian is to be computed at this value
    ///     * `ap_mat` *(output)* - the matrix to be used in constructing the (approximate) Jacobian as well as
    ///     the preconditioner.
    pub fn snes_set_jacobian_single_mat<F>(&mut self, user_f: F) -> Result<()>
    where
        F: FnMut(&SNES<'a, '_, '_>, &Vector<'a>, &mut Mat<'a, '_>) -> std::result::Result<(), DomainOrPetscError> + 'tl,
    {
        let closure_anchor = Box::new(user_f);

        let trampoline_data = Box::pin(SNESJacobianSingleTrampolineData { 
            world: self.world, user_f: closure_anchor });
        let _ = self.snes_jacobian_trampoline_data.take();

        unsafe extern "C" fn snes_jacobian_single_trampoline(snes_p: *mut petsc_raw::_p_SNES, vec_p: *mut petsc_raw::_p_Vec,
            mat1_p: *mut petsc_raw::_p_Mat, mat2_p: *mut petsc_raw::_p_Mat, ctx: *mut ::std::os::raw::c_void) -> petsc_raw::PetscErrorCode
        {
            // This assert should always be true based on how we constructed this wrapper
            assert!(mat1_p == mat2_p || mat2_p == 0 as *mut _);

            // SAFETY: read `snes_function_trampoline` safety
            let trampoline_data: Pin<&mut SNESJacobianSingleTrampolineData> = std::mem::transmute(ctx);

            // We don't want to drop anything, we are just using this to turn pointers 
            // of the underlining types (i.e. *mut petsc_raw::_p_SNES) into references
            // of the rust wrapper types.
            // If `Mat` ever has optional parameters, they MUST be dropped manually.
            // SAFETY: even though snes is mut and thus we can set optional parameters, we don't
            // as we dont expose the mut to the user closure, we only use it with `set_jacobian_domain_error`
            let mut snes = ManuallyDrop::new(SNES::new(trampoline_data.world, snes_p));

            let mut dm_p = MaybeUninit::uninit();
            let ierr = petsc_raw::SNESGetDM(snes_p, dm_p.as_mut_ptr());
            if ierr != 0 { let _ = chkerrq!(trampoline_data.world, ierr); return ierr; }
            let mut dm = DM::new(trampoline_data.world, dm_p.assume_init());
            let ierr = DM::set_inner_values_for_readonly(&mut dm);
            if ierr != 0 { return ierr; }
            snes.dm = Some(dm); // Note, because snes is not dropped, snes.dm wont be either

            let x = ManuallyDrop::new(Vector { world: trampoline_data.world, vec_p: vec_p });
            let mut a_mat = ManuallyDrop::new(Mat::new(trampoline_data.world, mat1_p));
            
            (trampoline_data.get_mut().user_f)(&snes, &x, &mut a_mat)
                .map_or_else(|err| match err {
                    DomainOrPetscError::DomainErr => {
                        let perr = snes.set_jacobian_domain_error();
                        match perr {
                            Ok(_) => 0,
                            Err(perr) => perr.kind as i32
                        }
                    },
                    DomainOrPetscError::PetscErr(perr) => perr.kind as i32
                }, |_| 0)
        }

        let ierr = unsafe { petsc_raw::DMSNESSetJacobian(
            self.dm_p, Some(snes_jacobian_single_trampoline), 
            std::mem::transmute(trampoline_data.as_ref())) };
        unsafe { chkerrq!(self.world, ierr) }?;
        
        self.snes_jacobian_trampoline_data = Some(SNESJacobianTrampolineData::SingleMat(trampoline_data));

        Ok(())
    }

    /// Set [`SNES`] Jacobian evaluation function, with the jacobian and the preconditioner being different.
    ///
    /// Note, [`SNES::set_jacobian()`] is normally used, but it calls this function internally
    /// because the user context is actually associated with the DM. This makes the interface consistent
    /// regardless of whether the user interacts with a DM or not. If DM took a more central role at some
    /// later date, this could become the primary method of setting the Jacobian. 
    ///
    /// # Parameters
    ///
    /// * `user_f` - A closure used to convey the Jacobian evaluation routine.
    ///     * `snes` - the snes context
    ///     * `x` - input vector, the Jacobian is to be computed at this value
    ///     * `a_mat` *(output)* - the matrix that defines the (approximate) Jacobian.
    ///     * `p_mat` *(output)* - the matrix to be used in constructing the preconditioner.
    pub fn snes_set_jacobian<F>(&mut self, user_f: F) -> Result<()>
    where
        F: FnMut(&SNES<'a, '_, '_>, &Vector<'a>, &mut Mat<'a, '_>, &mut Mat<'a, '_>) -> std::result::Result<(), DomainOrPetscError> + 'tl,
    {
        let closure_anchor = Box::new(user_f);

        let trampoline_data = Box::pin(SNESJacobianDoubleTrampolineData { 
            world: self.world, user_f: closure_anchor });
        let _ = self.snes_jacobian_trampoline_data.take();

        unsafe extern "C" fn snes_jacobian_double_trampoline(snes_p: *mut petsc_raw::_p_SNES, vec_p: *mut petsc_raw::_p_Vec,
            mat1_p: *mut petsc_raw::_p_Mat, mat2_p: *mut petsc_raw::_p_Mat, ctx: *mut ::std::os::raw::c_void) -> petsc_raw::PetscErrorCode
        {
            // This assert should always be true based on how we constructed this wrapper
            assert_ne!(mat1_p, mat2_p);

            // SAFETY: read `snes_function_trampoline` safety
            let trampoline_data: Pin<&mut SNESJacobianDoubleTrampolineData> = std::mem::transmute(ctx);

            // We don't want to drop anything, we are just using this to turn pointers 
            // of the underlining types (i.e. *mut petsc_raw::_p_SNES) into references.
            // If `Mat` ever has optional parameters, they MUST be dropped manually.
            // SAFETY: even though snes is mut and thus we can set optional parameters, we don't
            // as we dont expose the mut to the user closure, we only use it with `set_jacobian_domain_error`
            let mut snes = ManuallyDrop::new(SNES::new(trampoline_data.world, snes_p));

            let mut dm_p = MaybeUninit::uninit();
            let ierr = petsc_raw::SNESGetDM(snes_p, dm_p.as_mut_ptr());
            if ierr != 0 { let _ = chkerrq!(trampoline_data.world, ierr); return ierr; }
            let mut dm = DM::new(trampoline_data.world, dm_p.assume_init());
            let ierr = DM::set_inner_values_for_readonly(&mut dm);
            if ierr != 0 { return ierr; }
            snes.dm = Some(dm); // Note, because snes is not dropped, snes.dm wont be either

            let x = ManuallyDrop::new(Vector { world: trampoline_data.world, vec_p: vec_p });
            let mut a_mat = ManuallyDrop::new(Mat::new(trampoline_data.world, mat1_p));
            let mut p_mat = ManuallyDrop::new(Mat::new(trampoline_data.world, mat2_p));
            
            (trampoline_data.get_mut().user_f)(&snes, &x, &mut a_mat, &mut p_mat)
                .map_or_else(|err| match err {
                    DomainOrPetscError::DomainErr => {
                        let perr = snes.set_jacobian_domain_error();
                        match perr {
                            Ok(_) => 0,
                            Err(perr) => perr.kind as i32
                        }
                    },
                    DomainOrPetscError::PetscErr(perr) => perr.kind as i32
                }, |_| 0)
        }

        let ierr = unsafe { petsc_raw::DMSNESSetJacobian(
            self.dm_p, Some(snes_jacobian_double_trampoline), 
            std::mem::transmute(trampoline_data.as_ref())) };
        unsafe { chkerrq!(self.world, ierr) }?;
        
        self.snes_jacobian_trampoline_data = Some(SNESJacobianTrampolineData::DoubleMat(trampoline_data));

        Ok(())
    }

    /// Set [`KSP`] matrix evaluation function.
    ///
    /// Note, [`KSP::set_compute_operators()`] is normally used, but it calls this function internally
    /// because the user context is actually associated with the DM. This makes the interface consistent
    /// regardless of whether the user interacts with a DM or not. If DM took a more central role at some
    /// later date, this could become the primary method of setting the matrix.
    ///
    /// # Parameters
    ///
    /// * `user_f` - A closure used to convey the routine to compute the operators.
    ///     * `ksp` - the ksp context
    ///     * `a_mat` *(output)* - the linear operator
    ///     * `p_mat` *(output)* - preconditioning matrix
    pub fn ksp_set_compute_operators<F>(&mut self, user_f: F) -> Result<()>
    where
        F: FnMut(&KSP<'a, '_, '_>, &mut Mat<'a, '_>, &mut Mat<'a, '_>) -> Result<()> + 'tl,
    {
        // TODO: look at how rsmpi did the trampoline stuff:
        // https://github.com/rsmpi/rsmpi/blob/82e1d357/src/collective.rs#L1684
        // They used libffi, that could be a safer way to do it.

        let closure_anchor = Box::new(user_f);

        let trampoline_data = Box::pin(KSPComputeOperatorsTrampolineData { 
            world: self.world, user_f: closure_anchor });

        // drop old trampoline_data
        let _ = self.ksp_compute_operators_trampoline_data.take();

        unsafe extern "C" fn ksp_compute_operators_trampoline(ksp_p: *mut petsc_raw::_p_KSP, mat1_p: *mut petsc_raw::_p_Mat,
            mat2_p: *mut petsc_raw::_p_Mat, ctx: *mut std::os::raw::c_void) -> petsc_raw::PetscErrorCode
        {
            // SAFETY: We construct ctx to be a Pin<Box<KSPComputeOperatorsTrampolineData>> but pass it in as a *void
            // Box<T> is equivalent to *T (or &T) for ffi. Because the KSP owns the closure we can make sure
            // everything in it (and the closure its self) lives for at least as long as this function can be
            // called.
            // We don't construct a Box<> because we dont want to drop anything
            let trampoline_data: Pin<&mut KSPComputeOperatorsTrampolineData> = std::mem::transmute(ctx);

            // We don't want to drop anything, we are just using this to turn pointers 
            // of the underlining types (i.e. *mut petsc_raw::_p_SNES) into references
            // of the rust wrapper types.
            let mut ksp = ManuallyDrop::new(KSP::new(trampoline_data.world, ksp_p));

            let mut dm_p = MaybeUninit::uninit();
            let ierr = petsc_raw::KSPGetDM(ksp_p, dm_p.as_mut_ptr());
            if ierr != 0 { let _ = chkerrq!(trampoline_data.world, ierr); return ierr; }
            let mut dm = DM::new(trampoline_data.world, dm_p.assume_init());
            let ierr = DM::set_inner_values_for_readonly(&mut dm);
            if ierr != 0 { return ierr; }
            ksp.dm = Some(dm); // Note, because ksp is not dropped, ksp.dm wont be either

            let mut a_mat = ManuallyDrop::new(Mat::new(trampoline_data.world, mat1_p));
            let mut p_mat = ManuallyDrop::new(Mat::new(trampoline_data.world, mat2_p));
            
            (trampoline_data.get_mut().user_f)(&ksp, &mut a_mat, &mut p_mat)
                .map_or_else(|err| err.kind as i32, |_| 0)
        }

        let ierr = unsafe { petsc_raw::DMKSPSetComputeOperators(
            self.dm_p, Some(ksp_compute_operators_trampoline), 
            std::mem::transmute(trampoline_data.as_ref())) }; // this will also erase the lifetimes
        unsafe { chkerrq!(self.world, ierr) }?;
        
        self.ksp_compute_operators_trampoline_data = Some(trampoline_data);

        Ok(())
    }

    /// Set [`KSP`] right hand side evaluation function.
    ///
    /// Note, [`KSP::set_compute_rhs()`] is normally used, but it calls this function internally
    /// because the user context is actually associated with the DM. This makes the interface consistent
    /// regardless of whether the user interacts with a DM or not. If DM took a more central role at some
    /// later date, this could become the primary method of setting the rhs.
    ///
    /// # Parameters
    ///
    /// * `user_f` - A closure used to convey the routine to compute the the right hand side of the linear system
    ///     * `ksp` - the ksp context
    ///     * `b` *(output)* - right hand side of linear system
    pub fn ksp_set_compute_rhs<F>(&mut self, user_f: F) -> Result<()>
    where
        F: FnMut(&KSP<'a, '_, '_>, &mut Vector<'a>) -> Result<()> + 'tl,
    {
        // TODO: look at how rsmpi did the trampoline stuff:
        // https://github.com/rsmpi/rsmpi/blob/82e1d357/src/collective.rs#L1684
        // They used libffi, that could be a safer way to do it.

        let closure_anchor = Box::new(user_f);

        let trampoline_data = Box::pin(KSPComputeRHSTrampolineData { 
            world: self.world, user_f: closure_anchor });

        // drop old trampoline_data
        let _ = self.ksp_compute_rhs_trampoline_data.take();

        unsafe extern "C" fn ksp_compute_rhs_trampoline(ksp_p: *mut petsc_raw::_p_KSP, vec_p: *mut petsc_raw::_p_Vec,
            ctx: *mut std::os::raw::c_void) -> petsc_raw::PetscErrorCode
        {

            // SAFETY: read `ksp_compute_operators_single_trampoline` safety
            let trampoline_data: Pin<&mut KSPComputeRHSTrampolineData> = std::mem::transmute(ctx);

            // We don't want to drop anything, we are just using this to turn pointers 
            // of the underlining types (i.e. *mut petsc_raw::_p_SNES) into references
            // of the rust wrapper types.
            let mut ksp = ManuallyDrop::new(KSP::new(trampoline_data.world, ksp_p));

            let mut dm_p = MaybeUninit::uninit();
            let ierr = petsc_raw::KSPGetDM(ksp_p, dm_p.as_mut_ptr());
            if ierr != 0 { let _ = chkerrq!(trampoline_data.world, ierr); return ierr; }
            let mut dm = DM::new(trampoline_data.world, dm_p.assume_init());
            let ierr = DM::set_inner_values_for_readonly(&mut dm);
            if ierr != 0 { return ierr; }
            ksp.dm = Some(dm); // Note, because ksp is not dropped, ksp.dm wont be either

            let mut vec = ManuallyDrop::new(Vector { world: trampoline_data.world, vec_p });
            
            (trampoline_data.get_mut().user_f)(&ksp, &mut vec)
                .map_or_else(|err| err.kind as i32, |_| 0)
        }

        let ierr = unsafe { petsc_raw::DMKSPSetComputeRHS(
            self.dm_p, Some(ksp_compute_rhs_trampoline), 
            std::mem::transmute(trampoline_data.as_ref())) }; // this will also erase the lifetimes
        unsafe { chkerrq!(self.world, ierr) }?;
        
        self.ksp_compute_rhs_trampoline_data = Some(trampoline_data);

        Ok(())
    }
}

impl<'a> FEDisc<'a, '_> {
    /// Same as `FEDisc { ... }` but sets all optional params to `None`
    pub(crate) fn new(world: &'a UserCommunicator, fe_p: *mut petsc_raw::_p_PetscFE) -> Self {
        FEDisc { world, fe_p, space: None, dual_space: None }
    }

    /// Create a Field for basic FEM computation.
    ///
    /// # Parameters
    ///
    /// * `world` - The MPI comm world
    /// * `dim` - The spatial dimension
    /// * `nc` - The number of components
    /// * `is_simplex` - Flag for simplex reference cell, otherwise its a tensor product 
    /// * `prefix` - The options prefix, or `None`
    /// * `qorder` - The quadrature order or `None` to use PetscSpace polynomial degree 
    pub fn create_default<'pl>(world: &'a UserCommunicator, dim: PetscInt, nc: PetscInt, is_simplex: bool,
        prefix: impl Into<Option<&'pl str>>, qorder: impl Into<Option<PetscInt>>) -> Result<Self>
    {
        let cstring = prefix.into().map(|p| CString::new(p).expect("`CString::new` failed"));
        let mut fe_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::PetscFECreateDefault(world.as_raw(), dim, nc, is_simplex.into(),
            cstring.as_ref().map_or(std::ptr::null(), |p| p.as_ptr()), qorder.into().unwrap_or(petsc_raw::PETSC_DETERMINE),
            fe_p.as_mut_ptr()) };
        unsafe { chkerrq!(world, ierr) }?;

        Ok(FEDisc::new(world, unsafe { fe_p.assume_init() }))
    }

    /// Copy both volumetric and surface quadrature from `other`.
    pub fn copy_quadrature_from(&mut self, other: &FEDisc) -> Result<()> {
        let ierr = unsafe { petsc_raw::PetscFECopyQuadrature(other.fe_p, self.fe_p) };
        unsafe { chkerrq!(self.world, ierr) }
    }

    /// Determines whether a PETSc [`FEDisc`] is of a particular type.
    pub fn type_compare(&self, type_kind: FEDiscType) -> Result<bool> {
        self.type_compare_str(&type_kind.to_string())
    }
}

impl<'a> FVDisc<'a> {
    /// Same as `FVDisc { ... }` but sets all optional params to `None`
    pub(crate) fn new(world: &'a UserCommunicator, fv_p: *mut petsc_raw::_p_PetscFV) -> Self {
        FVDisc { world, fv_p }
    }
}

impl<'a, 'tl> DS<'a, 'tl> {
    /// Same as `DS { ... }` but sets all optional params to `None`
    pub(crate) fn new(world: &'a UserCommunicator, ds_p: *mut petsc_raw::_p_PetscDS) -> Self {
        DS { world, ds_p, residual_trampoline_data: None, jacobian_trampoline_data: None,
            exact_soln_trampoline_data: None,
            boundary_trampoline_data: None, }
    }

    /// Gets a boundary condition to the model 
    ///
    /// Unlike for the C API, this will not return the function pointers.
    ///
    /// This API is for PETSc `v3.16-dev.0`
    ///
    /// # Outputs (in order)
    ///
    /// * `wf` - The PetscWeakForm holding the pointwise functions
    /// * `type` - The type of condition, e.g. [`DM_BC_ESSENTIAL`](petsc_raw::DMBoundaryConditionType::DM_BC_ESSENTIAL)/
    /// [`DM_BC_ESSENTIAL_FIELD`](petsc_raw::DMBoundaryConditionType::DM_BC_ESSENTIAL_FIELD) (Dirichlet), or
    /// [`DM_BC_NATURAL`](petsc_raw::DMBoundaryConditionType::DM_BC_NATURAL) (Neumann).
    /// * `name` - The BC name
    /// * `label` - The label defining constrained points
    /// * `values` -  An array of ids for constrained points 
    /// * `field` - The field to constrain 
    /// * `comps` - An array of constrained component numbers
    #[cfg(any(petsc_version_3_16_dev, doc))]
    pub fn get_boundary_info(&self, bd: PetscInt) -> Result<(WeakForm<'a>, DMBoundaryConditionType, &str, DMLabel<'a>, &[PetscInt], PetscInt, &[PetscInt])> {
        let mut wf_p = ::std::mem::MaybeUninit::uninit();
        let mut bct = ::std::mem::MaybeUninit::uninit();
        let mut name_cs = ::std::mem::MaybeUninit::uninit();
        let mut dml_p = ::std::mem::MaybeUninit::uninit();
        let mut nv = ::std::mem::MaybeUninit::uninit();
        let mut values_p = ::std::mem::MaybeUninit::uninit();
        let mut field = ::std::mem::MaybeUninit::uninit();
        let mut nc = ::std::mem::MaybeUninit::uninit();
        let mut comps_p = ::std::mem::MaybeUninit::uninit();

        let ierr = unsafe { petsc_raw::PetscDSGetBoundary(self.ds_p, bd, wf_p.as_mut_ptr(), bct.as_mut_ptr(),
            name_cs.as_mut_ptr(), dml_p.as_mut_ptr(), nv.as_mut_ptr(), values_p.as_mut_ptr(),
            field.as_mut_ptr(), nc.as_mut_ptr(), comps_p.as_mut_ptr(), std::ptr::null_mut(),
            std::ptr::null_mut(), std::ptr::null_mut() ) };
        unsafe { chkerrq!(self.world, ierr) }?;

        // TODO: should we return refrences to wf and label?
        let mut wf = WeakForm { world: self.world, wf_p: unsafe { wf_p.assume_init() } };
        unsafe { wf.reference()?; }
        let mut label = DMLabel { world: self.world, dml_p: unsafe { dml_p.assume_init() } };
        unsafe { label.reference()?; }
        let name = unsafe { CStr::from_ptr(name_cs.assume_init()) }.to_str().unwrap();
        let values = unsafe { slice::from_raw_parts(values_p.assume_init(), nv.assume_init() as usize) };
        let comps = unsafe { slice::from_raw_parts(comps_p.assume_init(), nc.assume_init() as usize) };

        Ok((wf, unsafe { bct.assume_init() }, name, label, values, unsafe { field.assume_init() }, comps))
    }

    /// Gets a boundary condition to the model 
    ///
    /// Unlike for the C API, this will not return the function pointers.
    ///
    /// This API is for PETSc `v3.15`
    ///
    /// # Outputs (in order)
    ///
    /// * `type` - The type of condition, e.g. [`DM_BC_ESSENTIAL`](petsc_raw::DMBoundaryConditionType::DM_BC_ESSENTIAL)/
    /// [`DM_BC_ESSENTIAL_FIELD`](petsc_raw::DMBoundaryConditionType::DM_BC_ESSENTIAL_FIELD) (Dirichlet), or
    /// [`DM_BC_NATURAL`](petsc_raw::DMBoundaryConditionType::DM_BC_NATURAL) (Neumann).
    /// * `name` - The BC name
    /// * `labelname` - The label defining constrained points
    /// * `field` - The field to constrain
    /// * `comps` - An array of constrained component numbers
    /// * `ids` -  An array of ids for constrained points
    #[cfg(any(petsc_version_3_15, doc))]
    pub fn get_boundary_info(&self, bd: PetscInt) -> Result<(DMBoundaryConditionType, &str, &str, PetscInt, &[PetscInt], &[PetscInt])> {
        let mut bct = ::std::mem::MaybeUninit::uninit();
        let mut name_cs = ::std::mem::MaybeUninit::uninit();
        let mut label_cs = ::std::mem::MaybeUninit::uninit();
        let mut field = ::std::mem::MaybeUninit::uninit();
        let mut nc = ::std::mem::MaybeUninit::uninit();
        let mut comps_p = ::std::mem::MaybeUninit::uninit();
        let mut nids = ::std::mem::MaybeUninit::uninit();
        let mut ids_p = ::std::mem::MaybeUninit::uninit();

        let ierr = unsafe { petsc_raw::PetscDSGetBoundary(self.ds_p, bd, bct.as_mut_ptr(),
            name_cs.as_mut_ptr(), label_cs.as_mut_ptr(), field.as_mut_ptr(), nc.as_mut_ptr(),
            comps_p.as_mut_ptr(),  std::ptr::null_mut(), std::ptr::null_mut(), nids.as_mut_ptr(),
            ids_p.as_mut_ptr(), std::ptr::null_mut() ) };
        unsafe { chkerrq!(self.world, ierr) }?;

        // TODO: should we return refrences to wf and label?
        let name = unsafe { CStr::from_ptr(name_cs.assume_init()) }.to_str().unwrap();
        let label = unsafe { CStr::from_ptr(label_cs.assume_init()) }.to_str().unwrap();
        let comps = unsafe { slice::from_raw_parts(comps_p.assume_init(), nc.assume_init() as usize) };
        let ids = unsafe { slice::from_raw_parts(ids_p.assume_init(), nids.assume_init() as usize) };

        Ok((unsafe { bct.assume_init() }, name, label, unsafe { field.assume_init() }, comps, ids))
    }

    /// Set the array of constants passed to point functions
    pub fn set_constants(&mut self, consts: &[PetscInt]) -> Result<()> {
        // why does it take a `*mut _` for consts? It doesn't really
        // matter because it only does an array copy under the hood.
        let ierr = unsafe { petsc_raw::PetscDSSetConstants(self.ds_p, consts.len() as PetscInt, consts.as_ptr() as *mut _) };
        unsafe { chkerrq!(self.world, ierr) }
    }

    /// Set the pointwise exact solution function for a given test field
    ///
    /// # Parameters
    ///
    /// * `f` - The test field number 
    /// * `user_f` - solution function for the test fields 
    ///     * `dim` - the spatial dimension
    ///     * `t` - current time
    ///     * `x` - coordinates of the current point
    ///     * `nc` - the number of field components
    ///     * `u` *(output)* - the solution field evaluated at the current point
    pub fn set_exact_solution<F>(&mut self, f: PetscInt, user_f: F) -> Result<()>
    where
        F: Fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl
    {
        let closure_anchor = Box::new(user_f);

        let trampoline_data = Box::pin(DSExactSolutionTrampolineData { 
            user_f: closure_anchor });
        let _ = self.exact_soln_trampoline_data.take();

        unsafe extern "C" fn ds_exact_solution_trampoline(dim: PetscInt, time: PetscReal, x: *const PetscReal,
            nc: PetscInt, u: *mut petsc_raw::PetscScalar, ctx: *mut ::std::os::raw::c_void) -> petsc_raw::PetscErrorCode
        {
            let trampoline_data: Pin<&mut DSExactSolutionTrampolineData> = std::mem::transmute(ctx);

            let x_slice = slice::from_raw_parts(x, dim as usize);
            let u_slice = slice::from_raw_parts_mut(u as *mut _, nc as usize);
            
            (trampoline_data.get_mut().user_f)(dim, time, x_slice, nc, u_slice)
                .map_or_else(|err| err.kind as i32, |_| 0)
        }

        let ierr = unsafe { petsc_raw::PetscDSSetExactSolution(
            self.ds_p, f, Some(ds_exact_solution_trampoline), 
            std::mem::transmute(trampoline_data.as_ref())) };
        unsafe { chkerrq!(self.world, ierr) }?;
        
        self.exact_soln_trampoline_data = Some(trampoline_data);

        Ok(())
    }

    /// Add an essential boundary condition to the model. (WIP Wrapper function)
    ///
    /// In the C API you would call `PetscDSAddBoundary` with the type being `DM_BC_ESSENTIAL`.
    ///
    /// This API is for PETSc `v3.16-dev.0`
    ///
    /// # Parameters
    ///
    /// * `self` - The DM, with a PetscDS that matches the problem being constrained
    /// * `name` - The BC name
    /// * `label` - The label defining constrained points
    /// * `values` - An array of values for constrained points
    /// * `field` - The field to constrain
    /// * `comps` - An array of constrained component numbers
    /// * `bc_user_func` - A pointwise function giving boundary values
    ///
    /// ## `bc_user_func` Parameters
    ///
    /// * `dim` - the spatial dimension
    /// * `time` - the time of the current point
    /// * `x` - coordinates of the current point
    /// * `nc` - the number of field components
    /// * `bcval` *(output)* - output values at the current point
    #[cfg(any(petsc_version_3_16_dev, doc))]
    pub fn add_boundary_essential<F1>(&mut self, name: &str, label: &DMLabel, values: &[PetscInt],
        field: PetscInt, comps: &[PetscInt], bc_user_func: F1) -> Result<PetscInt>
    where
        F1: Fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
    {
        let mut bd = MaybeUninit::uninit();

        let raw_fn_wrapper = |ds_p, name, nc, comps, field, fn1, fn2, ctx| {
            unsafe { petsc_raw::PetscDSAddBoundary(
            ds_p, DMBoundaryConditionType::DM_BC_ESSENTIAL, name,
            label.dml_p, values.len() as PetscInt,
            values.as_ptr(), field, nc, comps,
            fn1, fn2, ctx, bd.as_mut_ptr()) }
        };

        // I would like to have this function and `add_boundary_essential_with_dt` be the same method with
        // the second user func being an option. However, the reason why we dont do this is because rust
        // will complain and say it doesn't know what type `F2` should be when you set it to `None`. This is
        // why we use explicit types here bellow instead of just `Option::None`. While this isn't a problem
        // here, i dont want to make the caller have to know they need to do this as it feels unnecessary.
        self.add_boundary_func_shared(name, field, comps, bc_user_func,
            Option::<fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()>>::None,
            raw_fn_wrapper)?;

        Ok(unsafe { bd.assume_init() })
    }

    /// Add an essential boundary condition to the model. (WIP Wrapper function)
    ///
    /// In the C API you would call `PetscDSAddBoundary` with the type being `DM_BC_ESSENTIAL`.
    ///
    /// This API is for PETSc `v3.16-dev.0`
    ///
    /// # Parameters
    ///
    /// * `self` - The DM, with a PetscDS that matches the problem being constrained
    /// * `name` - The BC name
    /// * `label` - The label defining constrained points
    /// * `values` - An array of values for constrained points
    /// * `field` - The field to constrain
    /// * `comps` - An array of constrained component numbers
    /// * `bc_user_func` - A pointwise function giving boundary values
    /// * `bc_user_func_t` - A pointwise function giving the time deriative of the boundary values
    ///
    /// ## `bc_user_func`/`bc_user_func_t` Parameters
    ///
    /// * `dim` - the spatial dimension
    /// * `time` - the time of the current point
    /// * `x` - coordinates of the current point
    /// * `nc` - the number of field components
    /// * `bcval` *(output)* - output values at the current point
    #[cfg(any(petsc_version_3_16_dev, doc))]
    pub fn add_boundary_essential_with_dt<F1, F2>(&mut self, name: &str, label: &DMLabel, values: &[PetscInt],
        field: PetscInt, comps: &[PetscInt], bc_user_func: F1, bc_user_func_t: F2) -> Result<PetscInt>
    where
        F1: Fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
        F2: Fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
    {
        let mut bd = MaybeUninit::uninit();

        let raw_fn_wrapper = |ds_p, name, nc, comps, field, fn1, fn2, ctx| {
            unsafe { petsc_raw::PetscDSAddBoundary(
            ds_p, DMBoundaryConditionType::DM_BC_ESSENTIAL, name,
            label.dml_p, values.len() as PetscInt,
            values.as_ptr(), field, nc, comps,
            fn1, fn2, ctx, bd.as_mut_ptr()) }
        };

        self.add_boundary_func_shared(name, field, comps, bc_user_func, Some(bc_user_func_t), raw_fn_wrapper)?;

        Ok(unsafe { bd.assume_init() })
    }

    /// Add an essential boundary condition to the model. (WIP Wrapper function)
    ///
    /// In the C API you would call `PetscDSAddBoundary` with the type being `DM_BC_ESSENTIAL`.
    ///
    /// This API is for PETSc `v3.15`
    ///
    /// # Parameters
    ///
    /// * `self` - The DM, with a PetscDS that matches the problem being constrained
    /// * `name` - The BC name
    /// * `labelname` - The label defining constrained points
    /// * `field` - The field to constrain
    /// * `comps` - An array of constrained component numbers
    /// * `ids` - An array of ids for constrained points
    /// * `bc_user_func` - A pointwise function giving boundary values
    ///
    /// ## `bc_user_func` Parameters
    ///
    /// * `dim` - the spatial dimension
    /// * `time` - the time of the current point
    /// * `x` - coordinates of the current point
    /// * `nc` - the number of field components
    /// * `bcval` *(output)* - output values at the current point
    #[cfg(any(petsc_version_3_15, doc))]
    pub fn add_boundary_essential<F1>(&mut self, name: &str, labelname: &str, field: PetscInt,
        comps: &[PetscInt], ids: &[PetscInt], bc_user_func: F1) -> Result<()>
    where
        F1: Fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
    {
        let labelname_cs = CString::new(labelname).expect("`CString::new` failed");

        let raw_fn_wrapper = |ds_p, name, nc, comps, field, fn1, fn2, ctx| {
            unsafe { petsc_raw::PetscDSAddBoundary(
            ds_p, DMBoundaryConditionType::DM_BC_ESSENTIAL, name,
            labelname_cs.as_ptr(), field, nc, comps, fn1, fn2,
            ids.len() as PetscInt, ids.as_ptr(), ctx) }
        };
        
        self.add_boundary_func_shared(name, field, comps, bc_user_func,
            Option::<fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()>>::None,
            raw_fn_wrapper)
    }

    /// Add an essential boundary condition to the model. (WIP Wrapper function)
    ///
    /// In the C API you would call `PetscDSAddBoundary` with the type being `DM_BC_ESSENTIAL`.
    ///
    /// This API is for PETSc `v3.15`
    ///
    /// # Parameters
    ///
    /// * `self` - The DM, with a PetscDS that matches the problem being constrained
    /// * `name` - The BC name
    /// * `labelname` - The label defining constrained points
    /// * `field` - The field to constrain
    /// * `comps` - An array of constrained component numbers
    /// * `ids` - An array of ids for constrained points
    /// * `bc_user_func` - A pointwise function giving boundary values
    /// * `bc_user_func_t` - A pointwise function giving the time deriative of the boundary values
    ///
    /// ## `bc_user_func`/`bc_user_func_t` Parameters
    ///
    /// * `dim` - the spatial dimension
    /// * `time` - the time of the current point
    /// * `x` - coordinates of the current point
    /// * `nc` - the number of field components
    /// * `bcval` *(output)* - output values at the current point
    #[cfg(any(petsc_version_3_15, doc))]
    pub fn add_boundary_essential_with_dt<F1, F2>(&mut self, name: &str, labelname: &str, field: PetscInt,
        comps: &[PetscInt], ids: &[PetscInt], bc_user_func: F1, bc_user_func_t: F2) -> Result<()>
    where
        F1: Fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
        F2: Fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
    {
        let labelname_cs = CString::new(labelname).expect("`CString::new` failed");

        let raw_fn_wrapper = |ds_p, name, nc, comps, field, fn1, fn2, ctx| {
            unsafe { petsc_raw::PetscDSAddBoundary(
            ds_p, DMBoundaryConditionType::DM_BC_ESSENTIAL, name,
            labelname_cs.as_ptr(), field, nc, comps, fn1, fn2,
            ids.len() as PetscInt, ids.as_ptr(), ctx) }
        };
        
        self.add_boundary_func_shared(name, field, comps, bc_user_func, Some(bc_user_func_t), raw_fn_wrapper)
    }

    /// Does all the heavy lifting for `add_boundary_essential` and `add_boundary_natural` independent
    /// of the version of `PetscDSAddBoundary`
    fn add_boundary_func_shared<F1, F2, F3>(&mut self, name: &str, field: PetscInt,
        comps: &[PetscInt], bc_user_func: F1, bc_user_func_t: Option<F2>, add_boundary_wrapper: F3) -> Result<()>
    where
        F1: Fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
        F2: Fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
        F3: FnOnce(*mut petsc_raw::_p_PetscDS, *const ::std::os::raw::c_char, PetscInt,
            *const PetscInt, PetscInt, Option<unsafe extern "C" fn()>, Option<unsafe extern "C" fn()>,
            *mut ::std::os::raw::c_void) -> petsc_raw::PetscErrorCode,
    {
        let name_cs = CString::new(name).expect("`CString::new` failed");

        let bc_user_func_t_is_some = bc_user_func_t.is_some();
        let closure_anchor1 = Box::new(bc_user_func);
        let closure_anchor2 = bc_user_func_t
            .map(|bc_user_func_t| -> Box<BCFuncDyn<'tl>> { 
                Box::new(bc_user_func_t) });

        let trampoline_data = Box::pin(DMBoundaryFuncTrampolineData { 
            user_f1: closure_anchor1, user_f2: closure_anchor2 });

        unsafe extern "C" fn bc_func_trampoline(dim: PetscInt, time: PetscReal, x: *const PetscReal,
            nc: PetscInt, bcval: *mut petsc_raw::PetscScalar, ctx: *mut ::std::os::raw::c_void) -> petsc_raw::PetscErrorCode
        {
            let trampoline_data: Pin<&mut DMBoundaryFuncTrampolineData> = std::mem::transmute(ctx);

            let x_slice = slice::from_raw_parts(x, dim as usize);
            let bcval_slice = slice::from_raw_parts_mut(bcval as *mut _, nc as usize);
            
            (trampoline_data.get_mut().user_f1)(dim, time, x_slice, nc, bcval_slice)
                .map_or_else(|err| err.kind as i32, |_| 0)
        }

        unsafe extern "C" fn bc_func_t_trampoline(dim: PetscInt, time: PetscReal, x: *const PetscReal,
            nc: PetscInt, bcval: *mut petsc_raw::PetscScalar, ctx: *mut ::std::os::raw::c_void) -> petsc_raw::PetscErrorCode
        {
            let trampoline_data: Pin<&mut DMBoundaryFuncTrampolineData> = std::mem::transmute(ctx);

            let x_slice = slice::from_raw_parts(x, dim as usize);
            let bcval_slice = slice::from_raw_parts_mut(bcval as *mut _, nc as usize);
            
            (trampoline_data.get_mut().user_f2.as_mut().unwrap())(dim, time, x_slice, nc, bcval_slice)
                .map_or_else(|err| err.kind as i32, |_| 0)
        }

        // When it comes to function pointers, casting in NOT your friend. It causes the pointers 
        // to be curropted. You MUST be VERY explicit with the types. All in all, the following
        // two declarations will get the function pointers in options correctly (I think). Once
        // we have them, we can them manipulate them how we see fit.
        let bc_func_essential_trampoline_fn_ptr: ::std::option::Option<
            unsafe extern "C" fn(arg1: PetscInt, arg2: PetscReal, arg3: *const PetscReal, arg4: PetscInt,
                arg5: *mut petsc_raw::PetscScalar, arg6: *mut ::std::os::raw::c_void, ) -> petsc_raw::PetscErrorCode, >
            = Some(bc_func_trampoline);

        let mut bc_func_t_essential_trampoline_fn_ptr: ::std::option::Option<
            unsafe extern "C" fn(arg1: PetscInt, arg2: PetscReal, arg3: *const PetscReal, arg4: PetscInt,
                arg5: *mut petsc_raw::PetscScalar, arg6: *mut ::std::os::raw::c_void, ) -> petsc_raw::PetscErrorCode, >
            = Some(bc_func_t_trampoline);

        if !bc_user_func_t_is_some {
            bc_func_t_essential_trampoline_fn_ptr = None;
        }

        let ierr = add_boundary_wrapper(
            self.ds_p, name_cs.as_ptr(), comps.len() as PetscInt, comps.as_ptr(),
            field, unsafe { std::mem::transmute(bc_func_essential_trampoline_fn_ptr) },
            unsafe { std::mem::transmute(bc_func_t_essential_trampoline_fn_ptr) }, 
            unsafe { std::mem::transmute(trampoline_data.as_ref()) }
        );
        unsafe { chkerrq!(self.world, ierr) }?;
        
        if let Some(ref mut boundary_trampoline_data_vec) = self.boundary_trampoline_data {
            boundary_trampoline_data_vec.push(DMBoundaryTrampolineData::BCFunc(trampoline_data));
        } else {
            self.boundary_trampoline_data = Some(vec![DMBoundaryTrampolineData::BCFunc(trampoline_data)]);
        }

        Ok(())
    }

    /// Add a natural boundary condition to the model. (WIP Wrapper function)
    ///
    /// In the C API you would call `PetscDSAddBoundary` with the type being `DM_BC_NATURAL`.
    ///
    /// This API is for PETSc `v3.16-dev.0`
    ///
    /// # Parameters
    ///
    /// * `self` - The DM, with a PetscDS that matches the problem being constrained
    /// * `name` - The BC name
    /// * `label` - The label defining constrained points
    /// * `values` - An array of values for constrained points
    /// * `field` - The field to constrain
    /// * `comps` - An array of constrained component numbers
    /// * `bc_user_func` - A pointwise function giving boundary values
    ///
    /// # Example
    ///
    /// Look at [`snes-ex12`](https://gitlab.com/petsc/petsc-rs/-/blob/main/examples/snes/src/ex12.rs)
    #[cfg(any(petsc_version_3_16_dev, doc))]
    pub fn add_boundary_natural<F1>(&mut self, name: &str, label: &DMLabel, values: &[PetscInt],
        field: PetscInt, comps: &[PetscInt], bc_user_func: F1) -> Result<PetscInt>
    where
        F1: Fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
    {
        let mut bd = MaybeUninit::uninit();

        let raw_fn_wrapper = |ds_p, name, nc, comps, field, fn1, fn2, ctx| {
            unsafe { petsc_raw::PetscDSAddBoundary(
            ds_p, DMBoundaryConditionType::DM_BC_NATURAL, name,
            label.dml_p, values.len() as PetscInt,
            values.as_ptr(), field, nc, comps,
            fn1, fn2, ctx, bd.as_mut_ptr()) }
        };

        self.add_boundary_func_shared(name, field, comps, bc_user_func,
            Option::<fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()>>::None,
            raw_fn_wrapper)?;

        Ok(unsafe { bd.assume_init() })
    }

    /// Add a natural boundary condition to the model. (WIP Wrapper function)
    ///
    /// In the C API you would call `PetscDSAddBoundary` with the type being `DM_BC_NATURAL`.
    ///
    /// This API is for PETSc `v3.16-dev.0`
    ///
    /// # Parameters
    ///
    /// * `self` - The DM, with a PetscDS that matches the problem being constrained
    /// * `name` - The BC name
    /// * `label` - The label defining constrained points
    /// * `values` - An array of values for constrained points
    /// * `field` - The field to constrain
    /// * `comps` - An array of constrained component numbers
    /// * `bc_user_func` - A pointwise function giving boundary values
    /// * `bc_user_func_t` - A pointwise function giving the time deriative of the boundary values
    #[cfg(any(petsc_version_3_16_dev, doc))]
    pub fn add_boundary_natural_with_dt<F1, F2>(&mut self, name: &str, label: &DMLabel, values: &[PetscInt],
        field: PetscInt, comps: &[PetscInt], bc_user_func: F1, bc_user_func_t: F2) -> Result<PetscInt>
    where
        F1: Fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
        F2: Fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
    {
        let mut bd = MaybeUninit::uninit();

        let raw_fn_wrapper = |ds_p, name, nc, comps, field, fn1, fn2, ctx| {
            unsafe { petsc_raw::PetscDSAddBoundary(
            ds_p, DMBoundaryConditionType::DM_BC_NATURAL, name,
            label.dml_p, values.len() as PetscInt,
            values.as_ptr(), field, nc, comps,
            fn1, fn2, ctx, bd.as_mut_ptr()) }
        };

        self.add_boundary_func_shared(name, field, comps, bc_user_func, Some(bc_user_func_t), raw_fn_wrapper)?;

        Ok(unsafe { bd.assume_init() })
    }

    /// Add an essential boundary condition to the model. (WIP Wrapper function)
    ///
    /// In the C API you would call `PetscDSAddBoundary` with the type being `DM_BC_NATURAL`.
    ///
    /// This API is for PETSc `v3.15`
    ///
    /// # Parameters
    ///
    /// * `self` - The DM, with a PetscDS that matches the problem being constrained
    /// * `name` - The BC name
    /// * `labelname` - The label defining constrained points
    /// * `field` - The field to constrain
    /// * `comps` - An array of constrained component numbers
    /// * `ids` - An array of ids for constrained points
    #[cfg(any(petsc_version_3_15, doc))]
    pub fn add_boundary_natural<F1>(&mut self, name: &str, labelname: &str, field: PetscInt,
        comps: &[PetscInt], ids: &[PetscInt], bc_user_func: F1) -> Result<()>
    where
        F1: Fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
    {
        let labelname_cs = CString::new(labelname).expect("`CString::new` failed");

        let raw_fn_wrapper = |ds_p, name, nc, comps, field, fn1, fn2, ctx| {
            unsafe { petsc_raw::PetscDSAddBoundary(
            ds_p, DMBoundaryConditionType::DM_BC_NATURAL, name,
            labelname_cs.as_ptr(), field, nc, comps, fn1, fn2,
            ids.len() as PetscInt, ids.as_ptr(), ctx) }
        };
        
        self.add_boundary_func_shared(name, field, comps, bc_user_func,
            Option::<fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()>>::None,
            raw_fn_wrapper)
    }

    /// Add an essential boundary condition to the model. (WIP Wrapper function)
    ///
    /// In the C API you would call `PetscDSAddBoundary` with the type being `DM_BC_ESSENTIAL`.
    ///
    /// This API is for PETSc `v3.15`
    ///
    /// # Parameters
    ///
    /// * `self` - The DM, with a PetscDS that matches the problem being constrained
    /// * `name` - The BC name
    /// * `labelname` - The label defining constrained points
    /// * `field` - The field to constrain
    /// * `comps` - An array of constrained component numbers
    /// * `ids` - An array of ids for constrained points
    /// * `bc_user_func` - A pointwise function giving boundary values
    /// * `bc_user_func_t` - A pointwise function giving the time deriative of the boundary values
    #[cfg(any(petsc_version_3_15, doc))]
    pub fn add_boundary_natural_with_dt<F1, F2>(&mut self, name: &str, labelname: &str, field: PetscInt,
        comps: &[PetscInt], ids: &[PetscInt], bc_user_func: F1, bc_user_func_t: F2) -> Result<()>
    where
        F1: Fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
        F2: Fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
    {
        let labelname_cs = CString::new(labelname).expect("`CString::new` failed");

        let raw_fn_wrapper = |ds_p, name, nc, comps, field, fn1, fn2, ctx| {
            unsafe { petsc_raw::PetscDSAddBoundary(
            ds_p, DMBoundaryConditionType::DM_BC_NATURAL, name,
            labelname_cs.as_ptr(), field, nc, comps, fn1, fn2,
            ids.len() as PetscInt, ids.as_ptr(), ctx) }
        };
        
        self.add_boundary_func_shared(name, field, comps, bc_user_func, Some(bc_user_func_t), raw_fn_wrapper)
    }

    // TODO: should these be unsafe functions, or is having the callback be unsafe enough
    /// Add a essential or natural field boundary condition to the model. (WIP Wrapper function)
    ///
    /// In the C API you would call `PetscDSAddBoundary` with the `bctype` being `DM_BC_*_FIELD`.
    ///
    /// Note, the functions `bc_user_func` and `bc_user_func` can not be closures. And they
    /// Will take pointers instead of slices. 
    ///
    /// This API is for PETSc `v3.16-dev.0`
    ///
    /// # Parameters
    ///
    /// * `self` - The DM, with a PetscDS that matches the problem being constrained
    /// * `bctype` - The type of condition, e.g. DM_BC_ESSENTIAL_FIELD (Dirichlet),
    /// or DM_BC_NATURAL_FIELD (Neumann). Must end in field.
    /// * `name` - The BC name
    /// * `label` - The label defining constrained points
    /// * `values` - An array of values for constrained points
    /// * `field` - The field to constrain
    /// * `comps` - An array of constrained component numbers
    /// * `bc_user_func` - A pointwise function giving boundary values
    /// * `bc_user_func_t` - A pointwise function giving the time deriative of the boundary values, or `None`
    ///
    /// ## `bc_user_func`/`bc_user_func_t` Parameters
    ///
    /// * `dim` - the spatial dimension
    /// * `nf` - the number of fields
    /// * `u_off` - the offset into u[] and u_t[] for each field
    /// * `u_off_x` - the offset into u_x[] for each field
    /// * `u` - each field evaluated at the current point
    /// * `u_t` - the time derivative of each field evaluated at the current point
    /// * `u_x` - the gradient of each field evaluated at the current point
    /// * `a_off` - the offset into a[] and a_t[] for each auxiliary field
    /// * `a_off_x` - the offset into a_x[] for each auxiliary field
    /// * `a` - each auxiliary field evaluated at the current point
    /// * `a_t` - the time derivative of each auxiliary field evaluated at the current point
    /// * `a_x` - the gradient of auxiliary each field evaluated at the current point
    /// * `t` - current time
    /// * `x` - coordinates of the current point
    /// * `nc` - number of constant parameters
    /// * `consts` - constant parameters
    /// * `bcval` *(output)* - output values at the current point
    #[cfg(any(petsc_version_3_16_dev, doc))]
    pub fn add_boundary_field_raw(&mut self, bctype: DMBoundaryConditionType, name: &str, label: &DMLabel, values: &[PetscInt],
        field: PetscInt, comps: &[PetscInt],
        bc_user_func: impl Into<Option<unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar)>>,
        bc_user_func_t: impl Into<Option<unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar)>>) -> Result<PetscInt>
    {
        let mut bd = MaybeUninit::uninit();

        let raw_fn_wrapper = |ds_p, bctype, name, nc, comps, field, fn1, fn2, ctx| {
            unsafe { petsc_raw::PetscDSAddBoundary(
            ds_p, bctype, name, label.dml_p, values.len() as PetscInt,
            values.as_ptr(), field, nc, comps,
            fn1, fn2, ctx, bd.as_mut_ptr()) }
        };

        let bc_user_func = bc_user_func.into();
        let bc_user_func_t = bc_user_func_t.into();

        self.add_boundary_field_raw_shared(bctype, name, field, comps, bc_user_func, bc_user_func_t, raw_fn_wrapper)?;

        Ok(unsafe { bd.assume_init() })
    }

    /// Add a essential or natural field boundary condition to the model. (WIP Wrapper function)
    ///
    /// In the C API you would call `PetscDSAddBoundary` with the `bctype` being `DM_BC_*_FIELD`.
    ///
    /// This API is for PETSc `v3.15`
    ///
    /// # Parameters
    ///
    /// * `self` - The DM, with a PetscDS that matches the problem being constrained
    /// * `bctype` - The type of condition, e.g. DM_BC_ESSENTIAL_FIELD (Dirichlet),
    /// or DM_BC_NATURAL_FIELD (Neumann). Must end in field.
    /// * `name` - The BC name
    /// * `labelname` - The label defining constrained points
    /// * `field` - The field to constrain
    /// * `comps` - An array of constrained component numbers
    /// * `ids` - An array of ids for constrained points
    /// * `bc_user_func` - A pointwise function giving boundary values
    /// * `bc_user_func_t` - A pointwise function giving the time deriative of the boundary values, or `None`
    ///
    /// ## `bc_user_func`/`bc_user_func_t` Parameters
    ///
    /// * `dim` - the spatial dimension
    /// * `nf` - the number of fields
    /// * `u_off` - the offset into u[] and u_t[] for each field
    /// * `u_off_x` - the offset into u_x[] for each field
    /// * `u` - each field evaluated at the current point
    /// * `u_t` - the time derivative of each field evaluated at the current point
    /// * `u_x` - the gradient of each field evaluated at the current point
    /// * `a_off` - the offset into a[] and a_t[] for each auxiliary field
    /// * `a_off_x` - the offset into a_x[] for each auxiliary field
    /// * `a` - each auxiliary field evaluated at the current point
    /// * `a_t` - the time derivative of each auxiliary field evaluated at the current point
    /// * `a_x` - the gradient of auxiliary each field evaluated at the current point
    /// * `t` - current time
    /// * `x` - coordinates of the current point
    /// * `nc` - number of constant parameters
    /// * `consts` - constant parameters
    /// * `bcval` *(output)* - output values at the current point
    #[cfg(any(petsc_version_3_15, doc))]
    pub fn add_boundary_field_raw(&mut self, bctype: DMBoundaryConditionType, name: &str, labelname: &str, field: PetscInt,
        comps: &[PetscInt], ids: &[PetscInt],
        bc_user_func: impl Into<Option<unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar)>>,
        bc_user_func_t: impl Into<Option<unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar)>>) -> Result<()>
    {
        let labelname_cs = CString::new(labelname).expect("`CString::new` failed");

        let bc_user_func = bc_user_func.into();
        let bc_user_func_t = bc_user_func_t.into();

        let raw_fn_wrapper = |ds_p, bctype, name, nc, comps, field, fn1, fn2, ctx| {
            unsafe { petsc_raw::PetscDSAddBoundary(
            ds_p, bctype, name, labelname_cs.as_ptr(), field, nc, comps,
            fn1, fn2, ids.len() as PetscInt, ids.as_ptr(), ctx) }
        };
        
        self.add_boundary_field_raw_shared(bctype, name, field, comps, bc_user_func, bc_user_func_t, raw_fn_wrapper)
    }
    
    /// Does all the heavy lifting for `add_boundary_field_raw` independent of the version of `PetscDSAddBoundary`
    fn add_boundary_field_raw_shared<F3>(&mut self, bctype: DMBoundaryConditionType, name: &str, field: PetscInt,
        comps: &[PetscInt],
        bc_user_func: Option<unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar)>,
        bc_user_func_t: Option<unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar)>,
        add_boundary_wrapper: F3) -> Result<()>
    where
        F3: FnOnce(*mut petsc_raw::_p_PetscDS, DMBoundaryConditionType, *const ::std::os::raw::c_char, PetscInt,
            *const PetscInt, PetscInt, Option<unsafe extern "C" fn()>, Option<unsafe extern "C" fn()>,
            *mut ::std::os::raw::c_void) -> petsc_raw::PetscErrorCode,
    {
        let name_cs = CString::new(name).expect("`CString::new` failed");

        let bctype = match bctype {
            DMBoundaryConditionType::DM_BC_ESSENTIAL_FIELD
                | DMBoundaryConditionType::DM_BC_NATURAL_FIELD => bctype,
            DMBoundaryConditionType::DM_BC_ESSENTIAL
                | DMBoundaryConditionType::DM_BC_NATURAL
                | DMBoundaryConditionType::DM_BC_ESSENTIAL_BD_FIELD
                => return seterrq!(self.world, PetscErrorKind::PETSC_ERR_USER_INPUT,
                    format!("DM::add_boundary_field does not support non field boundary conditions. You gave {:?}.", bctype)),
            // This is for `DMBoundaryConditionType::DM_BC_NATURAL_RIEMANN`
            _ => return seterrq!(self.world, PetscErrorKind::PETSC_ERR_USER_INPUT,
                    format!("Unknown boundary condition type given to DM::add_boundary_field. You gave {:?}.", bctype)),
        };
    
        let ierr = add_boundary_wrapper(
            self.ds_p, bctype, name_cs.as_ptr(), comps.len() as PetscInt, comps.as_ptr(),
            field, unsafe { std::mem::transmute(bc_user_func) },
            unsafe { std::mem::transmute(bc_user_func_t) }, 
            std::ptr::null_mut()
        );
        unsafe { chkerrq!(self.world, ierr) }?;
    
        Ok(())
    }

    /// Add a essential field boundary condition to the model. (WIP Wrapper function)
    ///
    /// In the C API you would call `PetscDSAddBoundary` with the `bctype` being `DM_BC_ESSENTIAL_BD_FIELD`.
    ///
    /// Note, the functions `bc_user_func` and `bc_user_func` can not be closures. And they
    /// Will take pointers instead of slices. 
    ///
    /// This API is for PETSc `v3.16-dev.0`
    ///
    /// # Parameters
    ///
    /// * `self` - The DM, with a PetscDS that matches the problem being constrained
    /// * `name` - The BC name
    /// * `label` - The label defining constrained points
    /// * `values` - An array of values for constrained points
    /// * `field` - The field to constrain
    /// * `comps` - An array of constrained component numbers
    /// * `bc_user_func` - A pointwise function giving boundary values
    /// * `bc_user_func_t` - A pointwise function giving the time deriative of the boundary values, or `None`
    ///
    /// ## `bc_user_func`/`bc_user_func_t` Parameters
    ///
    /// * `dim` - the spatial dimension
    /// * `nf` - the number of fields
    /// * `u_off` - the offset into u[] and u_t[] for each field
    /// * `u_off_x` - the offset into u_x[] for each field
    /// * `u` - each field evaluated at the current point
    /// * `u_t` - the time derivative of each field evaluated at the current point
    /// * `u_x` - the gradient of each field evaluated at the current point
    /// * `a_off` - the offset into a[] and a_t[] for each auxiliary field
    /// * `a_off_x` - the offset into a_x[] for each auxiliary field
    /// * `a` - each auxiliary field evaluated at the current point
    /// * `a_t` - the time derivative of each auxiliary field evaluated at the current point
    /// * `a_x` - the gradient of auxiliary each field evaluated at the current point
    /// * `t` - current time
    /// * `x` - coordinates of the current point
    /// * `n` - facet normal at the current point
    /// * `nc` - number of constant parameters
    /// * `consts` - constant parameters
    /// * `bcval` *(output)* - output values at the current point
    #[cfg(any(petsc_version_3_16_dev, doc))]
    pub fn add_boundary_essential_bd_field_raw(&mut self, name: &str, label: &DMLabel, values: &[PetscInt],
        field: PetscInt, comps: &[PetscInt],
        bc_user_func: impl Into<Option<unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            PetscReal, *const PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar)>>,
        bc_user_func_t: impl Into<Option<unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            PetscReal, *const PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar)>>) -> Result<PetscInt>
    {
        let mut bd = MaybeUninit::uninit();
        let name_cs = CString::new(name).expect("`CString::new` failed");

        let bc_user_func = bc_user_func.into();
        let bc_user_func_t = bc_user_func_t.into();

        let ierr = unsafe { petsc_raw::PetscDSAddBoundary(
            self.ds_p, DMBoundaryConditionType::DM_BC_ESSENTIAL_BD_FIELD, name_cs.as_ptr(),
            label.dml_p, values.len() as PetscInt,
            values.as_ptr(), field, comps.len() as PetscInt, comps.as_ptr(),
            std::mem::transmute(bc_user_func),
            std::mem::transmute(bc_user_func_t), 
            std::ptr::null_mut(), bd.as_mut_ptr()) };
        unsafe { chkerrq!(self.world, ierr) }?;

        Ok(unsafe { bd.assume_init() })
    }

    /// Add a essential field boundary condition to the model. (WIP Wrapper function)
    ///
    /// In the C API you would call `PetscDSAddBoundary` with the `bctype` being `DM_BC_ESSENTIAL_BD_FIELD`.
    ///
    /// Note, the functions `bc_user_func` and `bc_user_func` can not be closures. And they
    /// Will take pointers instead of slices. 
    ///
    /// This API is for PETSc `v3.15`
    ///
    /// # Parameters
    ///
    /// * `self` - The DM, with a PetscDS that matches the problem being constrained
    /// * `name` - The BC name
    /// * `label` - The label defining constrained points
    /// * `values` - An array of values for constrained points
    /// * `field` - The field to constrain
    /// * `comps` - An array of constrained component numbers
    /// * `bc_user_func` - A pointwise function giving boundary values
    /// * `bc_user_func_t` - A pointwise function giving the time deriative of the boundary values, or `None`
    ///
    /// ## `bc_user_func`/`bc_user_func_t` Parameters
    ///
    /// * `dim` - the spatial dimension
    /// * `nf` - the number of fields
    /// * `u_off` - the offset into u[] and u_t[] for each field
    /// * `u_off_x` - the offset into u_x[] for each field
    /// * `u` - each field evaluated at the current point
    /// * `u_t` - the time derivative of each field evaluated at the current point
    /// * `u_x` - the gradient of each field evaluated at the current point
    /// * `a_off` - the offset into a[] and a_t[] for each auxiliary field
    /// * `a_off_x` - the offset into a_x[] for each auxiliary field
    /// * `a` - each auxiliary field evaluated at the current point
    /// * `a_t` - the time derivative of each auxiliary field evaluated at the current point
    /// * `a_x` - the gradient of auxiliary each field evaluated at the current point
    /// * `t` - current time
    /// * `x` - coordinates of the current point
    /// * `n` - facet normal at the current point
    /// * `nc` - number of constant parameters
    /// * `consts` - constant parameters
    /// * `bcval` *(output)* - output values at the current point
    #[cfg(any(petsc_version_3_15, doc))]
    pub fn add_boundary_essential_bd_field_raw(&mut self, name: &str, labelname: &str, field: PetscInt,
        comps: &[PetscInt], ids: &[PetscInt],
        bc_user_func: impl Into<Option<unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            PetscReal, *const PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar)>>,
        bc_user_func_t: impl Into<Option<unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
            PetscReal, *const PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar)>>) -> Result<()>
    {
        let labelname_cs = CString::new(labelname).expect("`CString::new` failed");
        let name_cs = CString::new(name).expect("`CString::new` failed");

        let bc_user_func = bc_user_func.into();
        let bc_user_func_t = bc_user_func_t.into();
        
        let ierr = unsafe { petsc_raw::PetscDSAddBoundary(
            self.ds_p, DMBoundaryConditionType::DM_BC_ESSENTIAL_BD_FIELD, name_cs.as_ptr(),
            labelname_cs.as_ptr(), field, comps.len() as i32, comps.as_ptr(),
            std::mem::transmute(bc_user_func), std::mem::transmute(bc_user_func_t),
            ids.len() as PetscInt, ids.as_ptr(), std::ptr::null_mut()) };
        unsafe { chkerrq!(self.world, ierr) }
    }
}

impl<'a, 'tl> Deref for BorrowDM<'a, 'tl, '_> {
    type Target = DM<'a, 'tl>;

    fn deref(&self) -> &DM<'a, 'tl> {
        &self.owned_dm
    }
}

impl<'a, 'tl> DerefMut for BorrowDM<'a, 'tl, '_> {
    fn deref_mut(&mut self) -> &mut DM<'a, 'tl> {
        &mut self.owned_dm
    }
}

impl<'a, 'tl> Clone for DM<'a, 'tl> {
    /// Will clone the DM.
    ///
    /// This method is the exact same as [`DM::clone_result()`] but will unwrap on the [`Result`].
    ///
    /// Note, you can NOT clone a `DMPLEX`. This will panic.
    /// Instead use [`DM::clone_shallow()`] or [`DM::clone_unchecked()`].
    fn clone(&self) -> Self {
        self.clone_result().unwrap()
    }
}

// macro impls
impl<'a> DM<'a, '_> {
    wrap_simple_petsc_member_funcs! {
        DMSetFromOptions, pub set_from_options, takes mut, #[doc = "Sets various SNES and KSP parameters from user options."];
        DMSetUp, pub set_up, takes mut, #[doc = "Sets up the internal data structures for the later use of a nonlinear solver. This will be automatically called with [`SNES::solve()`](crate::snes::SNES::solve())."];
        DMGetDimension, pub get_dimension, output PetscInt, dim, #[doc = "Return the topological dimension of the DM"];
        DMDAGetInfo, pub da_get_info, output PetscInt, dim, output PetscInt, bm, output PetscInt, bn, output PetscInt, bp, output PetscInt, m,
            output PetscInt, n, output PetscInt, p, output PetscInt, dof, output PetscInt, s, output DMBoundaryType, bx, output DMBoundaryType, by,
            output DMBoundaryType, bz, output DMDAStencilType, st,
            #[doc = "Gets information about a given distributed array.\n\n\
            # Outputs (in order)\n\n\
            * `dim` - dimension of the distributed array (1, 2, or 3)\n\
            * `M, N, P` - global dimension in each direction of the array\n\
            * `m, n, p` - corresponding number of procs in each dimension\n\
            * `dof` - number of degrees of freedom per node\n\
            * `s` - stencil width\n * `bx,by,bz` - type of ghost nodes at boundary\n\
            * `st` - stencil type"];
        DMDAGetCorners, pub da_get_corners, output PetscInt, x, output PetscInt, y, output PetscInt, z, output PetscInt, m, output PetscInt, n, output PetscInt, p,
            #[doc = "Returns the global (x,y,z) indices of the lower left corner and size of the local region, excluding ghost points.\n\n\
            # Outputs (in order)\n\n\
            * `x,y,z` - the corner indices (where y and z are optional; these are used for 2D and 3D problems)\n\
            * `m,n,p` - widths in the corresponding directions (where n and p are optional; these are used for 2D and 3D problems)"];
        DMDAGetGhostCorners, pub da_get_ghost_corners, output PetscInt, x, output PetscInt, y, output PetscInt, z, output PetscInt, m, output PetscInt, n, output PetscInt, p,
            #[doc = "Returns the global (x,y,z) indices of the lower left corner and size of the local region, including ghost points.\n\n\
            # Outputs (in order)\n\n\
            * `x,y,z` - the corner indices (where y and z are optional; these are used for 2D and 3D problems)\n\
            * `m,n,p` - widths in the corresponding directions (where n and p are optional; these are used for 2D and 3D problems)"];
        // TODO: would it be nicer to have this take in a Range<PetscReal>? (but then we couldn't use the macro)
        DMDASetUniformCoordinates, pub da_set_uniform_coordinates, input PetscReal, x_min, input PetscReal, x_max, input PetscReal, y_min,
            input PetscReal, y_max, input PetscReal, z_min, input PetscReal, z_max, #[doc = "Sets a DMDA coordinates to be a uniform grid.\n\n\
            `y` and `z` values will be ignored for 1 and 2 dimensional problems."];
        DMCompositeGetNumberDM, pub composite_get_num_dms_petsc, output PetscInt, ndms, #[doc = "Get's the number of DM objects in the DMComposite representation."]; // TODO: remove this maybe, we can just use .len() on dm list
        DMPlexSetRefinementUniform, pub plex_set_refinement_uniform, input bool, refinement_uniform, takes mut, #[doc = "Set the flag for uniform refinement"];
        DMPlexIsSimplex, pub plex_is_simplex, output bool, flg, #[doc = "Is the first cell in this mesh a simplex?\n\n\
            Only avalable for PETSc `v3.16-dev.0`"] #[cfg(any(petsc_version_3_16_dev, doc))];
        DMGetNumFields, pub get_num_fields, output PetscInt, nf, #[doc = "Get the number of fields in the DM"];
        DMSetAuxiliaryVec, pub set_auxiliary_vec, input Option<&DMLabel<'a>>, label .as_raw, input PetscInt, value, input Vector<'a>, aux .as_raw consume .aux_vec, takes mut,
            #[doc = "Set the auxiliary vector for region specified by the given label and value.\n\nOnly avalable for PETSc `v3.16-dev.0`"] #[cfg(any(petsc_version_3_16_dev, doc))];
        DMPlexLabelComplete, pub plex_label_complete, input &mut DMLabel, label .as_raw, #[doc = "Starting with a label marking points on a surface, we add the transitive closure to the surface."]; 
    }
}

impl DS<'_, '_> {
    wrap_simple_petsc_member_funcs! {
        PetscDSGetNumBoundary, get_num_boundary, output PetscInt, nbd, #[doc = "TODO"];
    }
}

impl<'a, 'tl> FEDisc<'a, 'tl> {
    wrap_simple_petsc_member_funcs! {
        PetscFESetFromOptions, pub set_from_options, takes mut, #[doc = "sets parameters in a PetscFE from the options database"];
        PetscFESetUp, pub set_up, takes mut, #[doc = "Construct data structures for the PetscFE"];
        PetscFESetBasisSpace, pub set_basis_space, input Space<'a>, sp .as_raw consume .space, takes mut, #[doc = "Sets the [`Space`] used for approximation of the solution"];
        PetscFESetDualSpace, pub set_dual_space, input DualSpace<'a, 'tl>, sp .as_raw consume .dual_space, takes mut, #[doc = "Sets the [`Space`] used for approximation of the solution"];
        PetscFESetNumComponents, pub set_num_components, input PetscInt, nc, takes mut, #[doc = "Sets the number of components in the element"];
    }
}

impl_petsc_object_traits! {
    DM, dm_p, petsc_raw::_p_DM, DMView, DMDestroy, '_;
    DMLabel, dml_p, petsc_raw::_p_DMLabel, DMLabelView, DMLabelDestroy;
    FEDisc, fe_p, petsc_raw::_p_PetscFE, PetscFEView, PetscFEDestroy, '_;
    FVDisc, fv_p, petsc_raw::_p_PetscFV, PetscFVView, PetscFVDestroy;
    DMField, field_p, petsc_raw::_p_DMField, DMFieldView, DMFieldDestroy;
    DS, ds_p, petsc_raw::_p_PetscDS, PetscDSView, PetscDSDestroy, '_;
    WeakForm, wf_p, petsc_raw::_p_PetscWeakForm, PetscWeakFormView, PetscWeakFormDestroy;
}
