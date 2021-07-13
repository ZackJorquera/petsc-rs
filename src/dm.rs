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
use std::rc::Rc;
use crate::{
    Petsc,
    petsc_raw,
    Result,
    PetscAsRaw,
    PetscObject,
    PetscReal,
    PetscInt,
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
pub struct DM<'a> {
    pub(crate) world: &'a UserCommunicator,
    pub(crate) dm_p: *mut petsc_raw::_p_DM,

    composite_dms: Option<Vec<DM<'a>>>,
}

pub use crate::petsc_raw::DMBoundaryType;
pub use crate::petsc_raw::DMDAStencilType;
pub use crate::petsc_raw::DMTypeEnum as DMType;

impl<'a> Drop for DM<'a> {
    fn drop(&mut self) {
        let ierr = unsafe { petsc_raw::DMDestroy(&mut self.dm_p as *mut _) };
        let _ = Petsc::check_error(self.world, ierr); // TODO: should I unwrap or what idk?
    }
}

impl<'a> DM<'a> {
    /// Same as `DM { ... }` but sets all optional params to `None`
    pub(crate) fn new(world: &'a UserCommunicator, dm_p: *mut petsc_raw::_p_DM) -> Self {
        DM { world, dm_p, composite_dms: None }
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
        I: IntoIterator<Item = DM<'a>>
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
            Petsc::set_error(self.world, PetscErrorKind::PETSC_ERROR_ARG_WRONGSTATE,
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
            Petsc::set_error(self.world, PetscErrorKind::PETSC_ERROR_ARG_WRONGSTATE,
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
    ///     Petsc::set_error(petsc.world(), PetscErrorKind::PETSC_ERROR_WRONG_MPI_SIZE, 
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
    ///     Petsc::set_error(petsc.world(), PetscErrorKind::PETSC_ERROR_WRONG_MPI_SIZE, 
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
            Petsc::set_error(self.world, PetscErrorKind::PETSC_ERROR_ARG_INCOMP, 
                format!("Vector local size {} is not compatible with DMDA local sizes {} or {}\n",
                    local_size,xm*ym*zm*dof,gxm*gym*gzm*dof))?;
        }

        if dim > 3 || dim < 1 {
            Petsc::set_error(self.world, PetscErrorKind::PETSC_ERROR_ARG_CORRUPT, 
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
            Petsc::set_error(self.world, PetscErrorKind::PETSC_ERROR_ARG_INCOMP, 
                format!("Vector local size {} is not compatible with DMDA local sizes {} or {}\n",
                    local_size,xm*ym*zm*dof,gxm*gym*gzm*dof))?;
        }

        if dim > 3 || dim < 1 {
            Petsc::set_error(self.world, PetscErrorKind::PETSC_ERROR_ARG_CORRUPT, 
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
            Petsc::set_error(self.world, PetscErrorKind::PETSC_ERROR_ARG_CORRUPT, 
                format!("DMDA dimension not 1, 2, or 3, it is {}\n",dim))?;
        }

        let ierr = unsafe { petsc_raw::DMDAGetOwnershipRanges(self.dm_p, lx.as_mut_ptr(),
            if dim >= 2 { ly.as_mut_ptr() } else { std::ptr::null_mut() }, 
            if dim >= 3 { lz.as_mut_ptr() } else { std::ptr::null_mut() } ) };
        Petsc::check_error(self.world, ierr)?;

        // SAFETY: Petsc says these are arrays of length size
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
    pub fn composite_dms(&self) -> Result<&[DM<'a>]> {
        self.composite_dms.as_ref().map(|c| c.as_ref()).ok_or_else(|| 
            Petsc::set_error(self.world, PetscErrorKind::PETSC_ERROR_ARG_WRONGSTATE,
                format!("There are no composite dms set, line: {}", line!())).unwrap_err())
    }

    // pub fn composite_dms_mut(&mut self) -> Result<Vec<&mut DM<'a>>> {
    //     if let Some(c) = self.composite_dms.as_mut() {
    //         Ok(c.iter_mut().collect())
    //     } else {
    //         Petsc::set_error(self.world, PetscErrorKind::PETSC_ERROR_ARG_WRONGSTATE,
    //             format!("There are no composite dms set, line: {}", line!())).map(|_| unreachable!())
    //     }
    // }

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
            Petsc::set_error(self.world, PetscErrorKind::PETSC_ERROR_ARG_WRONGSTATE,
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
            Petsc::set_error(self.world, PetscErrorKind::PETSC_ERROR_ARG_WRONGSTATE,
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
            Petsc::set_error(self.world, PetscErrorKind::PETSC_ERROR_ARG_WRONGSTATE,
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
            Petsc::set_error(self.world, PetscErrorKind::PETSC_ERROR_ARG_WRONGSTATE,
                format!("There are no composite dms set, line: {}", line!())).map(|_| unreachable!())
        }
    }

    /// This is a WIP, i want to use this instead of doing `if let Some(c) = self.composite_dms.as_ref()`
    /// everywhere.
    fn try_get_composite_dms(&self) -> Result<Option<&Vec<DM<'a>>>> {
        let is_dm_comp = self.type_compare(petsc_raw::DMTYPE_TABLE[DMType::DMCOMPOSITE as usize])?;
        if is_dm_comp {
            Ok(self.composite_dms.as_ref())
        } else {
            Ok(None)
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
    pub(crate) unsafe fn set_inner_values(dm: &mut DM<'a>) -> petsc_raw::PetscErrorCode {
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
        } else {
            0
        }
    }
}

impl<'a> Clone for DM<'a> {
    fn clone(&self) -> Self {
        // TODO: the docs say this is a shallow clone. How should we deal with this for rust
        // (rust (and the caller) thinks/expects it is a deep clone)
        // TODO: Also what should we do for DM composite type, i get an error when DMClone calls
        // `DMGetDimension` and then `DMSetDimension` (the dim is -1).
        if self.type_compare(petsc_raw::DMTYPE_TABLE[DMType::DMCOMPOSITE as usize]).unwrap() {
            let c = self.try_get_composite_dms().unwrap().unwrap();
            DM::composite_create(self.world, c.iter().cloned()).unwrap()
        } else {
            let mut dm2_p = MaybeUninit::uninit();
            let ierr = unsafe { petsc_raw::DMClone(self.dm_p, dm2_p.as_mut_ptr()) };
            Petsc::check_error(self.world, ierr).unwrap();
            DM { world: self.world, dm_p: unsafe { dm2_p.assume_init() }, composite_dms: None }
        }
    }
}

// macro impls
impl<'a> DM<'a> {
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
    }
}

impl_petsc_object_traits! { DM, dm_p, petsc_raw::_p_DM }

impl_petsc_view_func!{ DM, DMView }
