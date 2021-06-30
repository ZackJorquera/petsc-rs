//! DM objects are used to manage communication between the algebraic structures in PETSc ([`Vector`] and [`Mat`])
//! and mesh data structures in PDE-based (or other) simulations. See, for example, [`DM::da_create_1d()`].
//!
//! The DMDA class encapsulates a Cartesian structured mesh, with interfaces for both topology and geometry.
//! It is capable of parallel refinement and coarsening. Some support for parallel redistribution is
//! available through the PCTELESCOPE object. A piecewise linear discretization is assumed for operations
//! which require this information.
//!
//! PETSc C API docs: <https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DM/index.html>
//!
//! Also: <https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DMDA/DMDA.html#DMDA>

use ndarray::{ArrayView, ArrayViewMut};

/// Abstract PETSc object that manages an abstract grid object and its interactions with the algebraic solvers
pub struct DM<'a> {
    world: &'a dyn Communicator,
    pub(crate) dm_p: *mut petsc_raw::_p_DM,
}

pub use crate::petsc_raw::DMBoundaryType;
pub use crate::petsc_raw::DMDAStencilType;
pub use crate::petsc_raw::DMTypeEnum as DMType;

use crate::prelude::*;

impl<'a> Drop for DM<'a> {
    fn drop(&mut self) {
        let ierr = unsafe { petsc_raw::DMDestroy(&mut self.dm_p as *mut _) };
        let _ = Petsc::check_error(self.world, ierr); // TODO: should I unwrap or what idk?
    }
}

impl<'a> DM<'a> {
    /// Same as `DM { ... }` but sets all optional params to `None`
    pub(crate) fn new(world: &'a dyn Communicator, dm_p: *mut petsc_raw::_p_DM) -> Self {
        DM { world, dm_p }
    }

    /// Creates an empty DM object. The type can then be set with [`DM::set_type()`].
    pub fn create(world: &'a dyn Communicator) -> Result<Self> {
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
    pub fn da_create_1d(world: &'a dyn Communicator, bx: DMBoundaryType, nx: PetscInt, dof: PetscInt, s: PetscInt, lx: Option<&[PetscInt]>) -> Result<Self> {
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
    pub fn da_create_2d(world: &'a dyn Communicator, bx: DMBoundaryType, by: DMBoundaryType, stencil_type: DMDAStencilType, 
        nx: PetscInt, ny: PetscInt, px: Option<PetscInt>, py: Option<PetscInt>, dof: PetscInt, s: PetscInt, lx: Option<&[PetscInt]>, ly: Option<&[PetscInt]>) -> Result<Self>
    {
        let px = px.unwrap_or(petsc_raw::PETSC_DECIDE_INTEGER);
        let py = py.unwrap_or(petsc_raw::PETSC_DECIDE_INTEGER);
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
    pub fn da_create_3d(world: &'a dyn Communicator, bx: DMBoundaryType, by: DMBoundaryType, bz: DMBoundaryType, stencil_type: DMDAStencilType, 
        nx: PetscInt, ny: PetscInt, nz: PetscInt, px: Option<PetscInt>, py: Option<PetscInt>, pz: Option<PetscInt>, dof: PetscInt, s: PetscInt,
        lx: Option<&[PetscInt]>, ly: Option<&[PetscInt]>, lz: Option<&[PetscInt]>) -> Result<Self>
    {
        let px = px.unwrap_or(petsc_raw::PETSC_DECIDE_INTEGER);
        let py = py.unwrap_or(petsc_raw::PETSC_DECIDE_INTEGER);
        let pz = pz.unwrap_or(petsc_raw::PETSC_DECIDE_INTEGER);
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
    ///
    /// Also, the underling array has a fortran contiguous layout (column major) whereas the C api swaps the order
    /// of the indexing (so essentially has row major but transposed). This means that in rust you will index the array
    /// normally, but for best performance (i.e., with caching) you should treat it as column major.
    ///
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
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
    /// let (gxs, gys, _gzs, gxm, gym, _gzm) = dm.da_get_corners()?;
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
    ///         rhs_array.slice(s![gxs..(gxs+gxm), gys..(gys+gym)]).dim());
    ///     assert_eq!(g_view.slice(s![.., ..]), rhs_array.slice(s![gxs..(gxs+gxm), gys..(gys+gym)]));
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
                .reversed_axes() }; // TODO: add comments (e.x. why `reversed_axes()` and stuff)
                                    // https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DMDA/DMDAVecGetArray.html#DMDAVecGetArray

        Ok(crate::vector::VectorView { vec, array: unsafe { array.assume_init() }, ndarray })
    }

    /// Returns a multi-dimension mutable view that shares data with the underlying vector and is indexed using
    /// the local dimensions.
    ///
    /// # Note
    ///
    /// The C api version of this, (`DMDAVecGetArray`), returns an array using the global dimensions by
    /// applying an offset to the arrays. This method does NOT do that, the view must be indexed starting at zero.
    ///
    /// Also, the underling array has a fortran contiguous layout (column major) whereas the C api swaps the order
    /// of the indexing (so essentially has row major but transposed). This means that in rust you will index the array
    /// normally, but for best performance (i.e., with caching) you should treat it as column major.
    ///
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
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
    /// let (gxs, gys, _gzs, gxm, gym, _gzm) = dm.da_get_corners()?;
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
}

// macro impls
impl<'a> DM<'a> {
    wrap_simple_petsc_member_funcs! {
        DMSetFromOptions, set_from_options, dm_p, takes mut, #[doc = "Sets various SNES and KSP parameters from user options."];
        DMSetUp, set_up, dm_p, takes mut, #[doc = "Sets up the internal data structures for the later use of a nonlinear solver. This will be automatically called with [`SNES::solve()`]."];
        DMGetDimension, get_dimension, dm_p, output PetscInt, dim, #[doc = "Return the topological dimension of the DM"];
        
        DMDAGetInfo, da_get_info, dm_p, output PetscInt, dim, output PetscInt, bm, output PetscInt, bn, output PetscInt, bp, output PetscInt, m,
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
        DMDAGetCorners, da_get_corners, dm_p, output PetscInt, x, output PetscInt, y, output PetscInt, z, output PetscInt, m, output PetscInt, n, output PetscInt, p,
            #[doc = "Returns the global (x,y,z) indices of the lower left corner and size of the local region, excluding ghost points.\n\n\
            # Outputs (in order)\n\n\
            * `x,y,z` - the corner indices (where y and z are optional; these are used for 2D and 3D problems)\n\
            * `m,n,p` - widths in the corresponding directions (where n and p are optional; these are used for 2D and 3D problems)"];
        DMDAGetGhostCorners, da_get_ghost_corners, dm_p, output PetscInt, x, output PetscInt, y, output PetscInt, z, output PetscInt, m, output PetscInt, n, output PetscInt, p,
            #[doc = "Returns the global (x,y,z) indices of the lower left corner and size of the local region, including ghost points.\n\n\
            # Outputs (in order)\n\n\
            * `x,y,z` - the corner indices (where y and z are optional; these are used for 2D and 3D problems)\n\
            * `m,n,p` - widths in the corresponding directions (where n and p are optional; these are used for 2D and 3D problems)"];
        // TODO: would it be nicer to have this take in a Range<PetscReal>? (then we couldn't use the macro)
        DMDASetUniformCoordinates, da_set_uniform_coordinates, dm_p, input PetscReal, x_min, input PetscReal, x_max, input PetscReal, y_min,
            input PetscReal, y_max, input PetscReal, z_min, input PetscReal, z_max, #[doc = "Sets a DMDA coordinates to be a uniform grid.\n\n\
            `y` and `z` values will be ignored for 1 and 2 dimensional problems."];
    }
}

impl_petsc_object_funcs!{ DM, dm_p }

impl_petsc_view_func!{ DM, dm_p, DMView }
