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

// TODO: use `PetscObjectTypeCompare` to make sure we are using the correct type of DM

use core::slice;
use std::mem::{MaybeUninit, ManuallyDrop};
use std::ffi::CString;
use std::pin::Pin;
use std::ptr::NonNull;
use std::rc::Rc;
use crate::{
    Petsc,
    petsc_raw,
    Result,
    PetscAsRaw,
    PetscObject,
    PetscReal,
    PetscInt,
    PetscScalar,
    PetscErrorKind,
    InsertMode,
    vector::{self, Vector, },
    mat::{Mat, },
    indexset::{IS, },
};
use mpi::topology::UserCommunicator;
use mpi::traits::*;

use ndarray::{ArrayView, ArrayViewMut};

/// Abstract PETSc object that manages an abstract grid object and its interactions with the algebraic solvers
pub struct DM<'a, 'tl> {
    pub(crate) world: &'a UserCommunicator,
    pub(crate) dm_p: *mut petsc_raw::_p_DM,

    composite_dms: Option<Vec<DM<'a, 'tl>>>,

    fields: Option<Vec<(Option<&'tl DMLabel<'a>>, Field<'a>)>>,

    boundary_trampoline_data: Option<Vec<DMBoundaryTrampolineData<'tl>>>,
}

/// Object which encapsulates a subset of the mesh from of a [`DM`]
pub struct DMLabel<'a> {
    pub(crate) world: &'a UserCommunicator,
    pub(crate) dml_p: *mut petsc_raw::_p_DMLabel,
}

/// The PetscFE class encapsulates a finite element discretization.
/// Each PetscFE object contains a PetscSpace, PetscDualSpace, and
/// DMPlex in the classic Ciarlet triple representation. 
pub struct Field<'a> {
    pub(crate) world: &'a UserCommunicator,
    pub(crate) fe_p: *mut petsc_raw::_p_PetscFE,
}

pub use crate::petsc_raw::DMBoundaryType;
pub use crate::petsc_raw::DMDAStencilType;
pub use crate::petsc_raw::DMTypeEnum as DMType;
pub use crate::petsc_raw::DMBoundaryConditionType;

enum DMBoundaryTrampolineData<'tl> {
    BCEssential(Pin<Box<DMBoundaryEssentialTrampolineData<'tl>>>),
    BCField(Pin<Box<DMBoundaryFieldTrampolineData<'tl>>>),
}

struct DMBoundaryEssentialTrampolineData<'tl> {
    user_f1: Box<BCEssentialFuncDyn<'tl>>,
    user_f2: Option<Box<BCEssentialFuncDyn<'tl>>>,
}

struct DMBoundaryFieldTrampolineData<'tl> {
    user_f1: Box<BCFieldFuncDyn<'tl>>,
    user_f2: Option<Box<BCFieldFuncDyn<'tl>>>,
}

// TODO: should i use trait aliases. It doesn't really matter, but the Fn types are long
// pub trait BCEssentialFuncWrapper<'tl>: FnMut(PetscInt, PetscReal, &[PetscReal],
//     PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl {}
// #[feature(trait_alias)]
// pub trait BCEssentialFunc<'tl> = FnMut(PetscInt, PetscReal, &[PetscReal],
//     PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl;
type BCEssentialFuncDyn<'tl> = dyn FnMut(PetscInt, PetscReal, &[PetscReal],
    PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl;
// #[feature(trait_alias)]
// pub trait BCFieldFunc<'tl> = FnMut(PetscInt, PetscInt, PetscInt,
//     &[PetscInt], &[PetscInt], &[PetscReal], &[PetscReal], &[PetscReal],
//     &[PetscInt], &[PetscInt], &[PetscReal], &[PetscReal], &[PetscReal],
//     PetscReal, &[PetscInt], PetscInt, &[PetscScalar], &mut [PetscScalar]) -> Result<()> + 'tl;
// pub trait BCFieldFunc<'tl>: FnMut(PetscInt, PetscInt, PetscInt,
//     &[PetscInt], &[PetscInt], &[PetscReal], &[PetscReal], &[PetscReal],
//     &[PetscInt], &[PetscInt], &[PetscReal], &[PetscReal], &[PetscReal],
//     PetscReal, &[PetscInt], PetscInt, &[PetscScalar], &mut [PetscScalar]) -> Result<()> + 'tl {}
type BCFieldFuncDyn<'tl> = dyn FnMut(PetscInt, PetscInt, PetscInt,
    &[PetscInt], &[PetscInt], &[PetscReal], &[PetscReal], &[PetscReal],
    &[PetscInt], &[PetscInt], &[PetscReal], &[PetscReal], &[PetscReal],
    PetscReal, &[PetscReal], PetscInt, &[PetscScalar], &mut [PetscScalar]) -> Result<()> + 'tl;

impl<'a> Drop for DM<'a, '_> {
    fn drop(&mut self) {
        let ierr = unsafe { petsc_raw::DMDestroy(&mut self.dm_p as *mut _) };
        let _ = Petsc::check_error(self.world, ierr); // TODO: should I unwrap or what idk?
    }
}

impl<'a> Drop for DMLabel<'a> {
    fn drop(&mut self) {
        let ierr = unsafe { petsc_raw::DMLabelDestroy(&mut self.dml_p as *mut _) };
        let _ = Petsc::check_error(self.world, ierr); // TODO: should I unwrap or what idk?
    }
}

impl<'a> Drop for Field<'a> {
    fn drop(&mut self) {
        let ierr = unsafe { petsc_raw::PetscFEDestroy(&mut self.fe_p as *mut _) };
        let _ = Petsc::check_error(self.world, ierr); // TODO: should I unwrap or what idk?
    }
}

impl<'a, 'tl> DM<'a, 'tl> {
    /// Same as `DM { ... }` but sets all optional params to `None`
    pub(crate) fn new(world: &'a UserCommunicator, dm_p: *mut petsc_raw::_p_DM) -> Self {
        DM { world, dm_p, composite_dms: None, boundary_trampoline_data: None, fields: None }
    }

    /// Creates an empty DM object. The type can then be set with [`DM::set_type()`].
    pub fn create(world: &'a UserCommunicator) -> Result<Self> {
        let mut dm_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::DMCreate(world.as_raw(), dm_p.as_mut_ptr()) };
        Petsc::check_error(world, ierr)?;

        Ok(DM::new(world, unsafe { dm_p.assume_init() }))
    }

    /// Builds a DM, for a particular DM implementation.
    pub fn set_type(&mut self, dm_type: DMType) -> Result<()> {
        // This could be use the macro probably 
        let option_str = petsc_raw::DMTYPE_TABLE[dm_type as usize];
        let cstring = CString::new(option_str).expect("`CString::new` failed");
        let ierr = unsafe { petsc_raw::DMSetType(self.dm_p, cstring.as_ptr()) };
        Petsc::check_error(self.world, ierr)
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
        Petsc::check_error(world, ierr)?;

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
        Petsc::check_error(world, ierr)?;

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
        Petsc::check_error(world, ierr)?;

        Ok(DM::new(world, unsafe { dm_p.assume_init() }))
    }

    /// Creates a vector packer, used to generate "composite" vectors made up of several subvectors.
    pub fn composite_create<I>(world: &'a UserCommunicator, dms: I) -> Result<Self>
    where
        I: IntoIterator<Item = Self>
    {
        let mut dm_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::DMCompositeCreate(world.as_raw(), dm_p.as_mut_ptr()) };
        Petsc::check_error(world, ierr)?;

        let mut dm = DM::new(world, unsafe { dm_p.assume_init() });

        let dms = dms.into_iter().collect::<Vec<_>>();
        let _num_dm = dms.iter().map(|this_dm| {
            let ierr = unsafe { petsc_raw::DMCompositeAddDM(dm.dm_p, this_dm.dm_p) };
            Petsc::check_error(dm.world, ierr).map(|_| 1)
        }).sum::<Result<PetscInt>>()?;

        dm.composite_dms = Some(dms);

        Ok(dm)
    }

    /// Creates a DMPlex object, which encapsulates an unstructured mesh,
    /// or CW complex, which can be expressed using a Hasse Diagram. 
    pub fn dm_plex_create(world: &'a UserCommunicator) -> Result<Self> {
        let mut dm_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::DMPlexCreate(world.as_raw(), dm_p.as_mut_ptr()) };
        Petsc::check_error(world, ierr)?;

        Ok(DM::new(world, unsafe { dm_p.assume_init() }))
    }

    /// Creates a global vector from a DM object
    pub fn create_global_vector(&self) -> Result<Vector<'a>> {
        let mut vec_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::DMCreateGlobalVector(self.dm_p, vec_p.as_mut_ptr()) };
        Petsc::check_error(self.world, ierr)?;
        
        Ok(Vector { world: self.world, vec_p: unsafe { vec_p.assume_init() } })
    }

    // TODO: what do these do? and is the world usage correct? Like do we use the
    // world of the DM or just a single processes from it? or does it not matter?
    /// Creates a local vector from a DM object
    pub fn create_local_vector(&self) -> Result<Vector<'a>> {
        let mut vec_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::DMCreateLocalVector(self.dm_p, vec_p.as_mut_ptr()) };
        Petsc::check_error(self.world, ierr)?;
        
        Ok(Vector { world: self.world, vec_p: unsafe { vec_p.assume_init() } })
    }

    /// Gets a PETSc vector that may be used with the DM local routines.
    ///
    /// This vector has spaces for the ghost values. 
    ///
    /// The vector values are NOT initialized and may have garbage in them, so you may need to zero them.
    /// This is intended to be used for vectors you need for a short time, like within a single function
    /// call. For vectors that you intend to keep around (for example in a C struct) or pass around large
    /// parts of your code you should use [`DM::create_local_vector()`]. 
    pub fn get_local_vector(&self) -> Result<vector::BorrowVectorMut<'a, '_>> {
        // Note, under the hood `DMGetLocalVector` uses multiple different work vector 
        // that it will give access to us. Once it runs out it starts using `DMCreateLocalVector`.
        // Therefor we don't need to worry about this being called multiple times and causing
        // problems. At least I think we don't (it feels like interior mutability so i wonder if
        // we should be using UnsafeCell for something).
        let mut vec_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::DMGetLocalVector(self.dm_p, vec_p.as_mut_ptr()) };
        Petsc::check_error(self.world, ierr)?;

        // We dont want to drop the vec through Vector::drop
        let vec = ManuallyDrop::new(Vector { world: self.world, vec_p: unsafe { vec_p.assume_init() } });
        
        Ok(vector::BorrowVectorMut::new(vec, Some(Box::new(move |borrow_vec| {
            let ierr = unsafe { petsc_raw::DMRestoreLocalVector(self.dm_p, &mut borrow_vec.vec_p as *mut _) };
            let _ = Petsc::check_error(borrow_vec.world, ierr); // TODO: should I unwrap ?
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
            Petsc::set_error(self.world, PetscErrorKind::PETSC_ERR_ARG_WRONGSTATE,
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
            Petsc::set_error(self.world, PetscErrorKind::PETSC_ERR_ARG_WRONGSTATE,
                format!("There are no composite dms set, line: {}", line!())).map(|_| unreachable!())
        }
    }

    /// Gets empty Jacobian for a DM
    ///
    /// This properly preallocates the number of nonzeros in the sparse
    /// matrix so you do not need to do it yourself. 
    ///
    /// For structured grid problems, when you call [`Mat::view_with()`] on this matrix it is
    /// displayed using the global natural ordering, NOT in the ordering used internally by PETSc.
    pub fn create_matrix(&self) -> Result<Mat<'a>> {
        let mut mat_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::DMCreateMatrix(self.dm_p, mat_p.as_mut_ptr()) };
        Petsc::check_error(self.world, ierr)?;
        
        Ok(Mat { world: self.world, mat_p: unsafe { mat_p.assume_init() } })
    }

    /// Updates global vectors from local vectors.
    pub fn global_to_local(&self, global: &Vector<'a>, mode: InsertMode, local: &mut Vector<'a>) -> Result<()> {
        let ierr = unsafe { petsc_raw::DMGlobalToLocalBegin(self.dm_p, global.vec_p, mode, local.vec_p) };
        Petsc::check_error(self.world, ierr)?;
        let ierr = unsafe { petsc_raw::DMGlobalToLocalEnd(self.dm_p, global.vec_p, mode, local.vec_p) };
        Petsc::check_error(self.world, ierr)
    }

    /// Updates local vectors from global vectors.
    ///
    /// In the [`ADD_VALUES`](InsertMode::ADD_VALUES) case you normally would zero the receiving vector
    /// before beginning this operation. [`INSERT_VALUES`](InsertMode::INSERT_VALUES) is not supported
    /// for DMDA, in that case simply compute the values directly into a global vector instead of a local one.
    pub fn local_to_global(&self, local: &Vector<'a>, mode: InsertMode, global: &mut Vector<'a>) -> Result<()> {
        let ierr = unsafe { petsc_raw::DMLocalToGlobalBegin(self.dm_p, local.vec_p, mode, global.vec_p) };
        Petsc::check_error(self.world, ierr)?;
        let ierr = unsafe { petsc_raw::DMLocalToGlobalEnd(self.dm_p, local.vec_p, mode, global.vec_p) };
        Petsc::check_error(self.world, ierr)
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
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # use mpi::traits::*;
    /// # use ndarray::{Dimension, array, s};
    /// # fn main() -> petsc_rs::Result<()> {
    /// # let petsc = Petsc::init_no_args()?;
    /// // note, cargo wont run tests with mpi so this will always be run with
    /// // a single processor, but this example will also work in a multiprocessor
    /// // comm world.
    ///
    /// // Note, right now this example only works when `PetscScalar` is `PetscReal`.
    /// // It will fail to compile if `PetscScalar` is `PetscComplex`.
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
    ///         .map(|i| (i, i as PetscReal)),
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
    ///     &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0][osr_usize.clone()]);
    ///
    /// // Ignoring the layout, the array functionally looks like the following.
    /// if petsc.world().size() == 1 {
    ///     let rhs_array = array![[0.0, 5.0], 
    ///                            [1.0, 6.0], 
    ///                            [2.0, 7.0],
    ///                            [3.0, 8.0],
    ///                            [4.0, 9.0]];
    ///     assert_eq!(g_view.slice(s![.., ..]).dim(), rhs_array.dim());
    ///     assert_eq!(g_view.slice(s![.., ..]), rhs_array);
    /// } else if petsc.world().size() == 2 {
    ///     let rhs_array = array![[0.0, 3.0], 
    ///                            [1.0, 4.0], 
    ///                            [2.0, 5.0],
    ///                            [6.0, 8.0],
    ///                            [7.0, 9.0]];
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
    /// # fn main() -> petsc_rs::Result<()> {
    /// # let petsc = Petsc::init_no_args()?;
    /// // note, cargo wont run tests with mpi so this will always be run with
    /// // a single processor, but this example will also work in two processor
    /// // comm world.
    ///
    /// // Note, right now this example only works when `PetscScalar` is `PetscReal`.
    /// // It will fail to compile if `PetscScalar` is `PetscComplex`.
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
    ///         .map(|i| (i, i as PetscReal)),
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
    pub fn da_vec_view<'b>(&self, vec: &'b Vector<'a>) -> Result<crate::vector::VectorView<'a, 'b>> {
        let (xs, ys, zs, xm, ym, zm) = self.da_get_corners()?;
        let (dim, _, _, _, _, _, _, dof, _, _, _, _, _) = self.da_get_info()?;
        let local_size = vec.get_local_size()?;

        let (_gxs, _gys, _gzs, gxm, gym, gzm) = if local_size == xm*ym*zm*dof { 
            (xs, ys, zs, xm, ym, zm)
        } else {
            self.da_get_ghost_corners()?
        };

        if local_size != gxm*gym*gzm*dof {
            Petsc::set_error(self.world, PetscErrorKind::PETSC_ERR_ARG_INCOMP, 
                format!("Vector local size {} is not compatible with DMDA local sizes {} or {}\n",
                    local_size,xm*ym*zm*dof,gxm*gym*gzm*dof))?;
        }

        if dim > 3 || dim < 1 {
            Petsc::set_error(self.world, PetscErrorKind::PETSC_ERR_ARG_CORRUPT, 
                format!("DMDA dimension not 1, 2, or 3, it is {}\n",dim))?;
        }

        let mut array = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::VecGetArrayRead(vec.vec_p, array.as_mut_ptr() as *mut _) };
        Petsc::check_error(vec.world, ierr)?;

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
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # use mpi::traits::*;
    /// # use ndarray::{Dimension, array, s};
    /// # fn main() -> petsc_rs::Result<()> {
    /// # let petsc = Petsc::init_no_args()?;
    /// // Note, cargo wont run tests with mpi so this will always be run with
    /// // a single processor, but this example will also work in a multiprocessor
    /// // comm world.
    ///
    /// // Note, right now this example only works when `PetscScalar` is `PetscReal`.
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
    /// global.set_all(0.0)?;
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
    ///     .for_each(|((i,j), v)| *v = (i*2+j) as PetscReal);
    ///
    /// let rhs_array = array![[0.0, 1.0], 
    ///                        [2.0, 3.0], 
    ///                        [4.0, 5.0],
    ///                        [6.0, 7.0],
    ///                        [8.0, 9.0]];
    /// assert_eq!(g_view.slice(s![.., ..]).dim(),
    ///     rhs_array.slice(s![gxs..(gxs+gxm), gys..(gys+gym)]).dim());
    /// assert_eq!(g_view.slice(s![.., ..]), rhs_array.slice(s![gxs..(gxs+gxm), gys..(gys+gym)]));
    /// # Ok(())
    /// # }
    /// ```
    pub fn da_vec_view_mut<'b>(&self, vec: &'b mut Vector<'a>) -> Result<crate::vector::VectorViewMut<'a, 'b>> {
        let (xs, yx, zs, xm, ym, zm) = self.da_get_corners()?;
        let (dim, _, _, _, _, _, _, dof, _, _, _, _, _) = self.da_get_info()?;
        let local_size = vec.get_local_size()?;

        let (_gxs, _gyx, _gzs, gxm, gym, gzm) = if local_size == xm*ym*zm*dof { 
            (xs, yx, zs, xm, ym, zm)
        } else {
            self.da_get_ghost_corners()?
        };

        if local_size != gxm*gym*gzm*dof {
            Petsc::set_error(self.world, PetscErrorKind::PETSC_ERR_ARG_INCOMP, 
                format!("Vector local size {} is not compatible with DMDA local sizes {} or {}\n",
                    local_size,xm*ym*zm*dof,gxm*gym*gzm*dof))?;
        }

        if dim > 3 || dim < 1 {
            Petsc::set_error(self.world, PetscErrorKind::PETSC_ERR_ARG_CORRUPT, 
                format!("DMDA dimension not 1, 2, or 3, it is {}\n",dim))?;
        }

        let mut array = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::VecGetArray(vec.vec_p, array.as_mut_ptr() as *mut _) };
        Petsc::check_error(vec.world, ierr)?;

        let dims_r = [gzm as usize, gym as usize, (gxm*dof) as usize];

        let ndarray = unsafe {
            ArrayViewMut::from_shape_ptr(ndarray::IxDyn(&dims_r[(3-dim as usize)..]), array.assume_init())
                .reversed_axes() };

        Ok(crate::vector::VectorViewMut { vec, array: unsafe { array.assume_init() }, ndarray })
    }

    /// Sets the names of individual field components in multicomponent vectors associated with a DMDA.
    ///
    /// Note, you must call [`DM::set_up()`] before you call this.
    ///
    /// # Parameters
    ///
    /// * `nf` - field number for the DMDA (0, 1, ... dof-1), where dof indicates the number of
    /// degrees of freedom per node within the DMDA.
    /// * `name` - the name of the field (component)
    pub fn da_set_feild_name<T: ToString>(&mut self, nf: PetscInt, name: T) -> crate::Result<()> {
        let name_cs = CString::new(name.to_string()).expect("`CString::new` failed");
        
        let ierr = unsafe { petsc_raw::DMDASetFieldName(self.dm_p, nf, name_cs.as_ptr()) };
        Petsc::check_error(self.world, ierr)
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
            Petsc::set_error(self.world, PetscErrorKind::PETSC_ERR_ARG_CORRUPT, 
                format!("DMDA dimension not 1, 2, or 3, it is {}\n",dim))?;
        }

        let ierr = unsafe { petsc_raw::DMDAGetOwnershipRanges(self.dm_p, lx.as_mut_ptr(),
            if dim >= 2 { ly.as_mut_ptr() } else { std::ptr::null_mut() }, 
            if dim >= 3 { lz.as_mut_ptr() } else { std::ptr::null_mut() } ) };
        Petsc::check_error(self.world, ierr)?;

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
            Petsc::set_error(self.world, PetscErrorKind::PETSC_ERR_ARG_WRONGSTATE,
                format!("There are no composite dms set, line: {}", line!())).unwrap_err())
    }

    // pub fn composite_dms_mut(&mut self) -> Result<Vec<&mut DM<'a>>> {
    //     if let Some(c) = self.composite_dms.as_mut() {
    //         Ok(c.iter_mut().collect())
    //     } else {
    //         Petsc::set_error(self.world, PetscErrorKind::PETSC_ERR_ARG_WRONGSTATE,
    //             format!("There are no composite dms set, line: {}", line!())).map(|_| unreachable!())
    //     }
    // }

    /// adds a DM vector to a DMComposite 
    pub fn composite_add_dm(&mut self, dm: DM<'a, 'tl>) -> Result<()> {
        let is_dm_comp = self.type_compare(petsc_raw::DMTYPE_TABLE[DMType::DMCOMPOSITE as usize])?;
        if is_dm_comp {
            let ierr = unsafe { petsc_raw::DMCompositeAddDM(self.dm_p, dm.dm_p) };
            Petsc::check_error(dm.world, ierr)?;

            if let Some(c) = self.composite_dms.as_mut() {
                c.push(dm);
            } else {
                self.composite_dms = Some(vec![dm]);
            }

            Ok(())
        } else {
            Petsc::set_error(self.world, PetscErrorKind::PETSC_ERR_ARG_WRONGSTATE,
                    format!("The DM is not a composite DM, line: {}", line!()))
        }
    }

    /// Scatters from a global packed vector into its individual local vectors 
    pub fn composite_scatter<'v, I>(&self, gvec: &'v Vector<'a>, lvecs: I) -> Result<()>
    where
        I: IntoIterator<Item = &'v mut Vector<'a>>,
    {
        if let Some(c) = self.composite_dms.as_ref() {
            let mut lvecs_p =  lvecs.into_iter().map(|v| v.vec_p).collect::<Vec<_>>();

            assert_eq!(lvecs_p.len(), c.len());
            let ierr = unsafe { petsc_raw::DMCompositeScatterArray(self.dm_p, gvec.vec_p, lvecs_p.as_mut_ptr()) };
            Petsc::check_error(self.world, ierr)
        } else {
            Petsc::set_error(self.world, PetscErrorKind::PETSC_ERR_ARG_WRONGSTATE,
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
        if let Some(c) = self.composite_dms.as_ref() {
            let wanted = (0..c.len() as PetscInt).collect::<Vec<_>>();
            let mut vec_ps = vec![std::ptr::null_mut(); c.len()]; // TODO: can we use MaybeUninit

            let ierr = unsafe { petsc_raw::DMCompositeGetAccessArray(self.dm_p, gvec.vec_p,
                c.len() as PetscInt, wanted.as_ptr(), vec_ps.as_mut_ptr()) };
            Petsc::check_error(self.world, ierr)?;

            let gvec_rc = Rc::new(&*gvec);
            Ok(vec_ps.into_iter().zip(wanted).map(move |(v_p, i)| {
                let vec = ManuallyDrop::new(Vector { world: self.world, vec_p: v_p });
                let gvec_rc = gvec_rc.clone();
                vector::BorrowVectorMut::new(vec, Some(Box::new(move |borrow_vec| {
                    let i = i;
                    let ierr = unsafe { petsc_raw::DMCompositeRestoreAccessArray(
                        self.dm_p, gvec_rc.vec_p, 1, std::slice::from_ref(&i).as_ptr(),
                        std::slice::from_mut(&mut borrow_vec.vec_p).as_mut_ptr()) };
                    let _ = Petsc::check_error(self.world, ierr); // TODO: should I unwrap ?
                })))
            }).collect())

        } else {
            Petsc::set_error(self.world, PetscErrorKind::PETSC_ERR_ARG_WRONGSTATE,
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
            Petsc::check_error(self.world, ierr)?;

            let is_slice = unsafe { slice::from_raw_parts(is_array_p.assume_init(), len) };

            let ret_vec = is_slice.iter().map(|is_p| IS { world: self.world, is_p: *is_p }).collect();

            let cs_fn_name = CString::new("composite_get_global_indexsets").expect("CString::new failed");
            let cs_file_name = CString::new("dm.rs").expect("CString::new failed");
            
            // Note, `PetscFree` is a macro around `PetscTrFree`
            let ierr = unsafe { (petsc_raw::PetscTrFree.unwrap())(is_array_p.assume_init() as *mut _,
                line!() as i32, cs_fn_name.as_ptr(), cs_file_name.as_ptr()) };
            Petsc::check_error(self.world, ierr)?;

            Ok(ret_vec)
        } else {
            Petsc::set_error(self.world, PetscErrorKind::PETSC_ERR_ARG_WRONGSTATE,
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
            Petsc::check_error(self.world, ierr)?;

            let is_slice = unsafe { slice::from_raw_parts(is_array_p.assume_init(), len) };

            let ret_vec = is_slice.iter().map(|is_p| IS { world: self.world, is_p: *is_p }).collect();

            let cs_fn_name = CString::new("composite_get_global_indexsets").expect("CString::new failed");
            let cs_file_name = CString::new("dm.rs").expect("CString::new failed");
            
            let ierr = unsafe { (petsc_raw::PetscTrFree.unwrap())(is_array_p.assume_init() as *mut _,
                line!() as i32, cs_fn_name.as_ptr(), cs_file_name.as_ptr()) };
            Petsc::check_error(self.world, ierr)?;

            Ok(ret_vec)
        } else {
            Petsc::set_error(self.world, PetscErrorKind::PETSC_ERR_ARG_WRONGSTATE,
                format!("There are no composite dms set, line: {}", line!())).map(|_| unreachable!())
        }
    }

    /// This is a WIP, i want to use this instead of doing `if let Some(c) = self.composite_dms.as_ref()`
    /// everywhere.
    fn try_get_composite_dms(&self) -> Result<Option<&Vec<Self>>> {
        let is_dm_comp = self.type_compare(petsc_raw::DMTYPE_TABLE[DMType::DMCOMPOSITE as usize])?;
        if is_dm_comp {
            Ok(self.composite_dms.as_ref())
        } else {
            Ok(None)
        }
    }

    /// Add the discretization object for the given DM field 
    ///
    /// Note, The label indicates the support of the field, or is `None` for the entire mesh.
    pub fn add_field(&mut self, label: impl Into<Option<&'tl DMLabel<'a>>>, field: Field<'a>) -> Result<()>
    {
        // TODO: what type does the dm need to be, if any?
        let is_correct_type = true; // self.type_compare(petsc_raw::DMTYPE_TABLE[DMType::DMCOMPOSITE as usize])?;
        if is_correct_type {
            let label: Option<&DMLabel> = label.into();
            let ierr = unsafe { petsc_raw::DMAddField(self.dm_p, label.as_ref().map_or(std::ptr::null_mut(), |l| l.dml_p), field.fe_p as *mut _) };
            Petsc::check_error(self.world, ierr)?;

            if let Some(f) = self.fields.as_mut() {
                f.push((label, field));
            } else {
                self.fields = Some(vec![(label, field)]);
            }

            Ok(())
        } else {
            Petsc::set_error(self.world, PetscErrorKind::PETSC_ERR_ARG_WRONGSTATE,
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
    pub(crate) unsafe fn set_inner_values(dm: &mut Self) -> petsc_raw::PetscErrorCode {
        if dm.type_compare(petsc_raw::DMTYPE_TABLE[DMType::DMCOMPOSITE as usize]).unwrap() {
            let len = dm.composite_get_num_dms_petsc().unwrap();
            let mut dms_p = vec![std::ptr::null_mut(); len as usize]; // TODO: use MaybeUninit if we can
            let ierr = petsc_raw::DMCompositeGetEntriesArray(dm.dm_p, dms_p.as_mut_ptr());
            if ierr != 0 { let _ = Petsc::check_error(dm.world, ierr); return ierr; }

            if let Some(mut old_dms) = dm.composite_dms.take() {
                while !old_dms.is_empty() {
                    let _ = ManuallyDrop::new(old_dms.pop().unwrap());
                }
            }

            dm.composite_dms = Some(dms_p.into_iter().map(|dm_p| {
                let mut this_dm = DM::new(dm.world, dm_p);
                let _ierr = DM::set_inner_values(&mut this_dm);
                //if ierr != 0 { return ierr; }
                this_dm
            }).collect::<Vec<_>>());
            0
        } else if dm.type_compare(petsc_raw::DMTYPE_TABLE[DMType::DMDA as usize]).unwrap() {
            0
        } else {
            todo!()
        }
    }

    /// Create a label of the given name if it does not already exist 
    pub fn create_label(&mut self, name: &str) -> Result<()> {
        let name_cs = CString::new(name).expect("`CString::new` failed");
        let ierr = unsafe { petsc_raw::DMCreateLabel(self.dm_p, name_cs.as_ptr()) };
        Petsc::check_error(self.world, ierr)
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
    pub fn get_label(&self, name: &str) -> Result<Option<DMLabel>> {
        let name_cs = CString::new(name).expect("`CString::new` failed");
        let mut dm_label = MaybeUninit::<*mut petsc_raw::_p_DMLabel>::uninit();
        let ierr = unsafe { petsc_raw::DMGetLabel(self.dm_p, name_cs.as_ptr(), dm_label.as_mut_ptr()) };
        Petsc::check_error(self.world, ierr)?;

        let dm_label = NonNull::new(unsafe { dm_label.assume_init() } );

        Ok(dm_label.map(|nn_dml_p| DMLabel { world: self.world, dml_p: nn_dml_p.as_ptr() }))
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
    /// * `bc_user_func_t` - A pointwise function giving the time deriative of the boundary values, or `None`
    ///
    /// ## `bc_user_func`/`bc_user_func_t` Parameters
    ///
    /// * `dim` - the spatial dimension
    /// * `time` - the time of the current point
    /// * `x` - coordinates of the current point
    /// * `nc` - ???????? TODO
    /// * `bcval` *(output)* - output values at the current point
    #[cfg(any(petsc_version_3_16_dev, doc))]
    pub fn add_boundary_essential<F1, F2>(&mut self, name: &str, label: &DMLabel, values: &[PetscInt],
        field: PetscInt, comps: &[PetscInt], bc_user_func: F1, bc_user_func_t: impl Into<Option<F2>>) -> Result<PetscInt>
    where
        F1: FnMut(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
        F2: FnMut(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
    {
        let mut bd = MaybeUninit::uninit();

        let raw_fn_wrapper = |dm_p, bctype, name, nc, comps, field, fn1, fn2, ctx| {
            unsafe { petsc_raw::DMAddBoundary(
            dm_p, bctype, name, label.dml_p, values.len() as PetscInt,
            values.as_ptr(), field, nc, comps,
            fn1, fn2, ctx, bd.as_mut_ptr()) }
        };

        self.add_boundary_essential_shared(name, field, comps, bc_user_func, bc_user_func_t, raw_fn_wrapper)?;

        Ok(unsafe { bd.assume_init() })
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
    /// * `bc_user_func_t` - A pointwise function giving the time deriative of the boundary values, or `None`
    ///
    /// ## `bc_user_func`/`bc_user_func_t` Parameters
    ///
    /// * `dim` - the spatial dimension
    /// * `time` - the time of the current point
    /// * `x` - coordinates of the current point
    /// * `nc` - ???????? TODO
    /// * `bcval` *(output)* - output values at the current point
    #[cfg(any(petsc_version_3_15, doc))]
    pub fn add_boundary_essential<F1, F2>(&mut self, name: &str, labelname: &str, field: PetscInt,
        comps: &[PetscInt], ids: &[PetscInt], bc_user_func: F1, bc_user_func_t: impl Into<Option<F2>>) -> Result<()>
    where
        F1: FnMut(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
        F2: FnMut(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
    {
        let labelname_cs = CString::new(labelname).expect("`CString::new` failed");

        let raw_fn_wrapper = |dm_p, bctype, name, nc, comps, field, fn1, fn2, ctx| {
            unsafe { petsc_raw::DMAddBoundary(
            dm_p, bctype, name, labelname_cs.as_ptr(), field, nc, comps,
            fn1, fn2, ids.len() as PetscInt, ids.as_ptr(), ctx) }
        };
        
        self.add_boundary_essential_shared(name, field, comps, bc_user_func, bc_user_func_t, raw_fn_wrapper)
    }

    /// Does all the heavy lifting for `add_boundary_essential` independent of the version of `DMAddBoundary`
    fn add_boundary_essential_shared<F1, F2, F3>(&mut self, name: &str, field: PetscInt,
        comps: &[PetscInt], bc_user_func: F1, bc_user_func_t: impl Into<Option<F2>>, add_boundary_wrapper: F3) -> Result<()>
    where
        F1: FnMut(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
        F2: FnMut(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> Result<()> + 'tl,
        F3: FnOnce(*mut petsc_raw::_p_DM, DMBoundaryConditionType, *const ::std::os::raw::c_char, PetscInt,
            *const PetscInt, PetscInt, Option<unsafe extern "C" fn()>, Option<unsafe extern "C" fn()>,
            *mut ::std::os::raw::c_void) -> petsc_raw::PetscErrorCode,
    {
        let name_cs = CString::new(name).expect("`CString::new` failed");

        let bc_user_func_t = bc_user_func_t.into();
        let bc_user_func_t_is_some = bc_user_func_t.is_some();
        let closure_anchor1 = Box::new(bc_user_func);
        let closure_anchor2 = bc_user_func_t
            .map(|bc_user_func_t| -> Box<BCEssentialFuncDyn<'tl>> { 
                Box::new(bc_user_func_t) });

        let trampoline_data = Box::pin(DMBoundaryEssentialTrampolineData { 
            user_f1: closure_anchor1, user_f2: closure_anchor2 });

        unsafe extern "C" fn bc_func_essential_trampoline(dim: PetscInt, time: PetscReal, x: *const PetscReal,
            nc: PetscInt, bcval: *mut PetscScalar, ctx: *mut ::std::os::raw::c_void) -> petsc_raw::PetscErrorCode
        {
            let trampoline_data: Pin<&mut DMBoundaryEssentialTrampolineData> = std::mem::transmute(ctx);

            // TODO: is dim/nc the correct len?
            let x_slice = slice::from_raw_parts(x, dim as usize);
            let bcval_slice = slice::from_raw_parts_mut(bcval, nc as usize);
            
            (trampoline_data.get_unchecked_mut().user_f1)(dim, time, x_slice, nc, bcval_slice)
                .map_or_else(|err| err.kind as i32, |_| 0)
        }

        unsafe extern "C" fn bc_func_t_essential_trampoline(dim: PetscInt, time: PetscReal, x: *const PetscReal,
            nc: PetscInt, bcval: *mut PetscScalar, ctx: *mut ::std::os::raw::c_void) -> petsc_raw::PetscErrorCode
        {
            let trampoline_data: Pin<&mut DMBoundaryEssentialTrampolineData> = std::mem::transmute(ctx);

            // TODO: is dim/nc the correct len?
            let x_slice = slice::from_raw_parts(x, dim as usize);
            let bcval_slice = slice::from_raw_parts_mut(bcval, nc as usize);
            
            (trampoline_data.get_unchecked_mut().user_f2.as_mut().unwrap())(dim, time, x_slice, nc, bcval_slice)
                .map_or_else(|err| err.kind as i32, |_| 0)
        }

        let bc_func_essential_trampoline_fn_ptr: Option<unsafe extern "C" fn()> = Some(unsafe { 
            *(&bc_func_essential_trampoline as *const _ as *const _) });

        let bc_func_t_essential_trampoline_fn_ptr: Option<unsafe extern "C" fn()> = if bc_user_func_t_is_some {
            Some(unsafe { 
                *(&bc_func_t_essential_trampoline as *const _ as *const _) })
        } else { None };

        let ierr = add_boundary_wrapper(
            self.dm_p, DMBoundaryConditionType::DM_BC_ESSENTIAL,
            name_cs.as_ptr(), comps.len() as PetscInt, comps.as_ptr(),
            field, bc_func_essential_trampoline_fn_ptr, bc_func_t_essential_trampoline_fn_ptr, 
            unsafe { std::mem::transmute(trampoline_data.as_ref()) }
        );
        Petsc::check_error(self.world, ierr)?;
        
        if let Some(ref mut boundary_trampoline_data_vec) = self.boundary_trampoline_data {
            boundary_trampoline_data_vec.push(DMBoundaryTrampolineData::BCEssential(trampoline_data));
        } else {
            self.boundary_trampoline_data = Some(vec![DMBoundaryTrampolineData::BCEssential(trampoline_data)]);
        }

        Ok(())
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
    #[cfg(any(petsc_version_3_16_dev, doc))]
    pub fn add_boundary_natural(&mut self, name: &str, label: &DMLabel, values: &[PetscInt],
        field: PetscInt, comps: &[PetscInt]) -> Result<PetscInt>
    {
        let mut bd = MaybeUninit::uninit();

        let name_cs = CString::new(name).expect("`CString::new` failed");
        
        let ierr = unsafe { petsc_raw::DMAddBoundary(
            self.dm_p, DMBoundaryConditionType::DM_BC_NATURAL, name_cs.as_ptr(),
            label.dml_p, values.len() as PetscInt,
            values.as_ptr(), field, comps.len() as PetscInt, comps.as_ptr(),
            None, None, std::ptr::null_mut(), bd.as_mut_ptr()) };
        Petsc::check_error(self.world, ierr)?;

        Ok(unsafe { bd.assume_init() })
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
    pub fn add_boundary_natural(&mut self, name: &str, labelname: &str, field: PetscInt,
        comps: &[PetscInt], ids: &[PetscInt]) -> Result<()>
    {
        let name_cs = CString::new(name).expect("`CString::new` failed");
        let labelname_cs = CString::new(labelname).expect("`CString::new` failed");

        let ierr = unsafe { petsc_raw::DMAddBoundary(
            self.dm_p,  DMBoundaryConditionType::DM_BC_NATURAL, name_cs.as_ptr(), labelname_cs.as_ptr(),
            field, comps.len() as PetscInt, comps.as_ptr(),
            None, None, ids.len() as PetscInt, ids.as_ptr(), std::ptr::null_mut()) };

        Petsc::check_error(self.world, ierr)
    }

    // TODO: im having a hard time finding documentation of any of this so im going to hold of on
    // implementing this any furthur. Also, it looks like the documentation is different from the
    // examples (which makes things iteresting). For now im going to follow whats in the examples.
    // Regardless, that means none of this code is at all safe or even correct (same goes for all
    // `add_boundary` code, but more so for `add_boundary_field`). I also dont know where the source
    // code calls the functions. A good place to look might be at the commit that changed this method
    // on the C side. They must have also touched the internals.
    /// Add a essential or natural field boundary condition to the model. (WIP Wrapper function)
    ///
    /// In the C API you would call `DMAddBoundary` with the `bctype` being `DM_BC_*_FIELD`.
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
    pub fn add_boundary_field<F1, F2>(&mut self, bctype: DMBoundaryConditionType, name: &str, label: &DMLabel, values: &[PetscInt],
        field: PetscInt, comps: &[PetscInt], bc_user_func: F1, bc_user_func_t: impl Into<Option<F2>>) -> Result<PetscInt>
    where
        F1: FnMut(PetscInt, PetscInt, PetscInt,
            &[PetscInt], &[PetscInt], &[PetscReal], &[PetscReal], &[PetscReal],
            &[PetscInt], &[PetscInt], &[PetscReal], &[PetscReal], &[PetscReal],
            PetscReal, &[PetscReal], PetscInt, &[PetscScalar], &mut [PetscScalar]) -> Result<()> + 'tl,
        F2: FnMut(PetscInt, PetscInt, PetscInt,
            &[PetscInt], &[PetscInt], &[PetscReal], &[PetscReal], &[PetscReal],
            &[PetscInt], &[PetscInt], &[PetscReal], &[PetscReal], &[PetscReal],
            PetscReal, &[PetscReal], PetscInt, &[PetscScalar], &mut [PetscScalar]) -> Result<()> + 'tl,
    {
        let mut bd = MaybeUninit::uninit();

        let raw_fn_wrapper = |dm_p, bctype, name, nc, comps, field, fn1, fn2, ctx| {
            unsafe { petsc_raw::DMAddBoundary(
            dm_p, bctype, name, label.dml_p, values.len() as PetscInt,
            values.as_ptr(), field, nc, comps,
            fn1, fn2, ctx, bd.as_mut_ptr()) }
        };

        self.add_boundary_field_shared(bctype, name, field, comps, bc_user_func, bc_user_func_t, raw_fn_wrapper)?;

        Ok(unsafe { bd.assume_init() })
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
    pub fn add_boundary_field<F1, F2>(&mut self, bctype: DMBoundaryConditionType, name: &str, labelname: &str, field: PetscInt,
        comps: &[PetscInt], ids: &[PetscInt], bc_user_func: F1, bc_user_func_t: impl Into<Option<F2>>) -> Result<()>
    where
        F1: FnMut(PetscInt, PetscInt, PetscInt,
            &[PetscInt], &[PetscInt], &[PetscReal], &[PetscReal], &[PetscReal],
            &[PetscInt], &[PetscInt], &[PetscReal], &[PetscReal], &[PetscReal],
            PetscReal, &[PetscReal], PetscInt, &[PetscScalar], &mut [PetscScalar]) -> Result<()> + 'tl,
        F2: FnMut(PetscInt, PetscInt, PetscInt,
            &[PetscInt], &[PetscInt], &[PetscReal], &[PetscReal], &[PetscReal],
            &[PetscInt], &[PetscInt], &[PetscReal], &[PetscReal], &[PetscReal],
            PetscReal, &[PetscReal], PetscInt, &[PetscScalar], &mut [PetscScalar]) -> Result<()> + 'tl,
    {
        let labelname_cs = CString::new(labelname).expect("`CString::new` failed");

        let raw_fn_wrapper = |dm_p, bctype, name, nc, comps, field, fn1, fn2, ctx| {
            unsafe { petsc_raw::DMAddBoundary(
            dm_p, bctype, name, labelname_cs.as_ptr(), field, nc, comps,
            fn1, fn2, ids.len() as PetscInt, ids.as_ptr(), ctx) }
        };
        
        self.add_boundary_field_shared(bctype, name, field, comps, bc_user_func, bc_user_func_t, raw_fn_wrapper)
    }
    
    /// Does all the heavy lifting for `add_boundary_field` independent of the version of `DMAddBoundary`
    fn add_boundary_field_shared<F1, F2, F3>(&mut self, bctype: DMBoundaryConditionType, name: &str, field: PetscInt,
        comps: &[PetscInt], bc_user_func: F1, bc_user_func_t: impl Into<Option<F2>>, add_boundary_wrapper: F3) -> Result<()>
    where
        F1: FnMut(PetscInt, PetscInt, PetscInt,
            &[PetscInt], &[PetscInt], &[PetscReal], &[PetscReal], &[PetscReal],
            &[PetscInt], &[PetscInt], &[PetscReal], &[PetscReal], &[PetscReal],
            PetscReal, &[PetscReal], PetscInt, &[PetscScalar], &mut [PetscScalar]) -> Result<()> + 'tl,
        F2: FnMut(PetscInt, PetscInt, PetscInt,
            &[PetscInt], &[PetscInt], &[PetscReal], &[PetscReal], &[PetscReal],
            &[PetscInt], &[PetscInt], &[PetscReal], &[PetscReal], &[PetscReal],
            PetscReal, &[PetscReal], PetscInt, &[PetscScalar], &mut [PetscScalar]) -> Result<()> + 'tl,
        F3: FnOnce(*mut petsc_raw::_p_DM, DMBoundaryConditionType, *const ::std::os::raw::c_char, PetscInt,
            *const PetscInt, PetscInt, Option<unsafe extern "C" fn()>, Option<unsafe extern "C" fn()>,
            *mut ::std::os::raw::c_void) -> petsc_raw::PetscErrorCode,
    {
        let name_cs = CString::new(name).expect("`CString::new` failed");

        let bctype = match bctype {
            bct @ (DMBoundaryConditionType::DM_BC_ESSENTIAL
                | DMBoundaryConditionType::DM_BC_NATURAL
                | DMBoundaryConditionType::DM_BC_NATURAL_RIEMANN)
                => return Petsc::set_error(self.world, PetscErrorKind::PETSC_ERR_USER_INPUT,
                    format!("DM::add_boundary_field does not support non field boundary conditions. You gave {:?}.", bct)),
            bct @ _ => bct 
        };
    
        let bc_user_func_t = bc_user_func_t.into();
        let bc_user_func_t_is_some = bc_user_func_t.is_some();
        let closure_anchor1 = Box::new(bc_user_func);
        let closure_anchor2 = bc_user_func_t
            .map(|bc_user_func_t| -> Box<BCFieldFuncDyn<'tl>>
                { Box::new(bc_user_func_t) });
    
        let trampoline_data = Box::pin(DMBoundaryFieldTrampolineData { 
            user_f1: closure_anchor1, user_f2: closure_anchor2 });
    
        // TODO: It is unclear from looking at the docs and the examples if this method takes a ctx value
        // if it doesn't then we will probably have to somehow grab them from the DM struct (or really
        // the _n_DSBoundary stuct in c). With out a ctx this will not work.
        unsafe extern "C" fn bc_func_field_trampoline(dim: PetscInt, nf: PetscInt, nf_aux: PetscInt,
            u_off: *const PetscInt, u_off_x: *const PetscInt, u: *const PetscScalar, u_t: *const PetscScalar, u_x: *const PetscScalar,
            a_off: *const PetscInt, a_off_x: *const PetscInt, a: *const PetscScalar, a_t: *const PetscScalar, a_x: *const PetscScalar,
            time: PetscReal, x: *const PetscReal, nc: PetscInt, consts: *const PetscScalar, bcval: *mut PetscScalar, ctx: *mut ::std::os::raw::c_void) -> petsc_raw::PetscErrorCode
        {
            let trampoline_data: Pin<&mut DMBoundaryFieldTrampolineData> = std::mem::transmute(ctx);
    
            // TODO: what is the correct len for these array (im just guessing)?
            let u_off_slice = slice::from_raw_parts(u_off, dim as usize);
            let u_off_x_slice = slice::from_raw_parts(u_off_x, dim as usize);
            let u_slice = slice::from_raw_parts(u, dim as usize);
            let u_t_slice = slice::from_raw_parts(u_t, dim as usize);
            let u_x_slice = slice::from_raw_parts(u_x, dim as usize);
            
            let a_off_slice = slice::from_raw_parts(a_off, dim as usize);
            let a_off_x_slice = slice::from_raw_parts(a_off_x, dim as usize);
            let a_slice = slice::from_raw_parts(a, dim as usize);
            let a_t_slice = slice::from_raw_parts(a_t, dim as usize);
            let a_x_slice = slice::from_raw_parts(a_x, dim as usize);

            let x_slice = slice::from_raw_parts(x, dim as usize);

            let consts_slice = slice::from_raw_parts(consts, dim as usize);
            
            let bcval_slice = slice::from_raw_parts_mut(bcval, nf as usize);
    
            (trampoline_data.get_unchecked_mut().user_f1)(dim, nf, nf_aux,
                u_off_slice, u_off_x_slice, u_slice, u_t_slice, u_x_slice,
                a_off_slice, a_off_x_slice, a_slice, a_t_slice, a_x_slice,
                time, x_slice, nc, consts_slice, bcval_slice)
                .map_or_else(|err| err.kind as i32, |_| 0)
        }

        unsafe extern "C" fn bc_func_t_field_trampoline(dim: PetscInt, nf: PetscInt, nf_aux: PetscInt,
            u_off: *const PetscInt, u_off_x: *const PetscInt, u: *const PetscScalar, u_t: *const PetscScalar, u_x: *const PetscScalar,
            a_off: *const PetscInt, a_off_x: *const PetscInt, a: *const PetscScalar, a_t: *const PetscScalar, a_x: *const PetscScalar,
            time: PetscReal, x: *const PetscReal, nc: PetscInt, consts: *const PetscScalar, bcval: *mut PetscScalar, ctx: *mut ::std::os::raw::c_void) -> petsc_raw::PetscErrorCode
        {
            let trampoline_data: Pin<&mut DMBoundaryFieldTrampolineData> = std::mem::transmute(ctx);
    
            // TODO: what is the correct len for these array (im just guessing)?
            let u_off_slice = slice::from_raw_parts(u_off, dim as usize);
            let u_off_x_slice = slice::from_raw_parts(u_off_x, dim as usize);
            let u_slice = slice::from_raw_parts(u, dim as usize);
            let u_t_slice = slice::from_raw_parts(u_t, dim as usize);
            let u_x_slice = slice::from_raw_parts(u_x, dim as usize);
            
            let a_off_slice = slice::from_raw_parts(a_off, dim as usize);
            let a_off_x_slice = slice::from_raw_parts(a_off_x, dim as usize);
            let a_slice = slice::from_raw_parts(a, dim as usize);
            let a_t_slice = slice::from_raw_parts(a_t, dim as usize);
            let a_x_slice = slice::from_raw_parts(a_x, dim as usize);

            let x_slice = slice::from_raw_parts(x, dim as usize);

            let consts_slice = slice::from_raw_parts(consts, dim as usize);
            
            let bcval_slice = slice::from_raw_parts_mut(bcval, nf as usize);
    
            (trampoline_data.get_unchecked_mut().user_f2.as_mut().unwrap())(dim, nf, nf_aux,
                u_off_slice, u_off_x_slice, u_slice, u_t_slice, u_x_slice,
                a_off_slice, a_off_x_slice, a_slice, a_t_slice, a_x_slice,
                time, x_slice, nc, consts_slice, bcval_slice)
                .map_or_else(|err| err.kind as i32, |_| 0)
        }
    
        let bc_func_field_trampoline_fn_ptr: Option<unsafe extern "C" fn()> = Some(unsafe { 
            *(&bc_func_field_trampoline as *const _ as *const _) });
    
        let bc_func_t_field_trampoline_fn_ptr: Option<unsafe extern "C" fn()> = if bc_user_func_t_is_some {
            Some(unsafe { 
                *(&bc_func_t_field_trampoline as *const _ as *const _) })
        } else { None };
    
        let ierr = add_boundary_wrapper(
            self.dm_p, bctype, name_cs.as_ptr(), comps.len() as PetscInt, comps.as_ptr(),
            field, bc_func_field_trampoline_fn_ptr, bc_func_t_field_trampoline_fn_ptr, 
            unsafe { std::mem::transmute(trampoline_data.as_ref()) }
        );
        Petsc::check_error(self.world, ierr)?;
    
        if let Some(ref mut boundary_trampoline_data_vec) = self.boundary_trampoline_data {
            boundary_trampoline_data_vec.push(DMBoundaryTrampolineData::BCField(trampoline_data));
        } else {
            self.boundary_trampoline_data = Some(vec![DMBoundaryTrampolineData::BCField(trampoline_data)]);
        }
    
        Ok(())
    }

}

impl<'a> Clone for DM<'a, '_> {
    fn clone(&self) -> Self {
        // TODO: the docs say this is a shallow clone. How should we deal with this for rust
        // (rust (and the caller) thinks/expects it is a deep clone)
        // TODO: Also what should we do for DM composite type, i get an error when DMClone calls
        // `DMGetDimension` and then `DMSetDimension` (the dim is -1).
        if self.boundary_trampoline_data.is_some(){
            todo!();
        }
        if self.type_compare(petsc_raw::DMTYPE_TABLE[DMType::DMCOMPOSITE as usize]).unwrap() {
            let c = self.try_get_composite_dms().unwrap().unwrap();
            DM::composite_create(self.world, c.iter().cloned()).unwrap()
        } else {
            let mut dm2_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::DMClone(self.dm_p, dm2_p.as_mut_ptr()) };
            Petsc::check_error(self.world, ierr).unwrap();
            DM::new(self.world, unsafe { dm2_p.assume_init() })
        }
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
        // TODO: would it be nicer to have this take in a Range<PetscReal>? (then we couldn't use the macro)
        DMDASetUniformCoordinates, pub da_set_uniform_coordinates, input PetscReal, x_min, input PetscReal, x_max, input PetscReal, y_min,
            input PetscReal, y_max, input PetscReal, z_min, input PetscReal, z_max, #[doc = "Sets a DMDA coordinates to be a uniform grid.\n\n\
            `y` and `z` values will be ignored for 1 and 2 dimensional problems."];
        DMCompositeGetNumberDM, pub composite_get_num_dms_petsc, output PetscInt, ndms, #[doc = "idk remove this maybe"];
        DMCreateDS, pub create_ds, takes mut, #[doc = "Create the discrete systems for the DM based upon the fields added to the DM\n\n\
            Note, If the label has a DS defined, it will be replaced. Otherwise, it will be added to the DM. "];
    }
}

impl_petsc_object_traits! { DM, dm_p, petsc_raw::_p_DM, '_ }

impl_petsc_view_func!{ DM, DMView, '_ }

impl_petsc_object_traits! { DMLabel, dml_p, petsc_raw::_p_DMLabel }

impl_petsc_view_func!{ DMLabel, DMLabelView }

impl_petsc_object_traits! { Field, fe_p, petsc_raw::_p_PetscFE }

impl_petsc_view_func!{ Field, PetscFEView }
