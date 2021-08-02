//! PETSc matrices (Mat objects) are used to store Jacobians and other
//! sparse matrices in PDE-based (or other) simulations.
//!
//! PETSc C API docs: <https://petsc.org/release/docs/manualpages/Mat/index.html>

use std::{ffi::CString, marker::PhantomData, ops::{Deref, DerefMut}, pin::Pin};
use std::mem::{MaybeUninit, ManuallyDrop};
use std::rc::Rc;
use crate::{
    Petsc,
    petsc_raw,
    Result,
    PetscAsRaw,
    PetscObject,
    PetscScalar,
    PetscReal,
    PetscInt,
    InsertMode,
    NormType,
    PetscErrorKind,
    vector::{Vector, },
    indexset::{IS, },
};
use mpi::topology::UserCommunicator;
use mpi::traits::*;
use seq_macro::seq;

/// Abstract PETSc matrix object used to manage all linear operators in PETSc, even those
/// without an explicit sparse representation (such as matrix-free operators).
pub struct Mat<'a, 'tl> {
    pub(crate) world: &'a UserCommunicator,
    pub(crate) mat_p: *mut petsc_raw::_p_Mat, // I could use Mat which is the same thing, but i think using a pointer is more clear

    shell_trampoline_data: Option<Pin<Box<MatShellTrampolineData<'a, 'tl>>>>,
}

/// A wrapper around [`Mat`] that is used when the [`Mat`] shouldn't be destroyed.
///
/// Gives mutable access to the underlining [`Mat`].
///
/// For example, it is used with [`Mat::get_local_sub_matrix_mut()`].
pub struct BorrowMatMut<'a, 'tl, 'bv> {
    owned_mat: ManuallyDrop<Mat<'a, 'tl>>,
    drop_func: Option<Box<dyn FnOnce(&mut Self) + 'bv>>,
    // do we need this phantom data?
    // also should 'bv be used for the closure
    pub(crate) _phantom: PhantomData<&'bv mut Mat<'a, 'tl>>,
}

/// A wrapper around [`Mat`] that is used when the [`Mat`] shouldn't be destroyed.
///
/// For example, it is used with [`Mat::get_local_sub_matrix_mut()`].
pub struct BorrowMat<'a, 'tl, 'bv> {
    owned_mat: ManuallyDrop<Mat<'a, 'tl>>,
    drop_func: Option<Box<dyn FnOnce(&mut Self) + 'bv>>,
    // do we need this phantom data?
    // also should 'bv be used for the closure
    pub(crate) _phantom: PhantomData<&'bv Mat<'a, 'tl>>,
}

impl<'a> Drop for Mat<'a, '_> {
    fn drop(&mut self) {
        let ierr = unsafe { petsc_raw::MatDestroy(&mut self.mat_p as *mut _) };
        let _ = Petsc::check_error(self.world, ierr); // TODO: should I unwrap or what idk?
    }
}

impl Drop for BorrowMatMut<'_, '_, '_> {
    fn drop(&mut self) {
        self.drop_func.take().map(|f| f(self));
    }
}

impl Drop for BorrowMat<'_, '_, '_> {
    fn drop(&mut self) {
        self.drop_func.take().map(|f| f(self));
    }
}

/// Abstract PETSc object that removes a null space from a vector, i.e. orthogonalizes the vector to a subspace.
pub struct NullSpace<'a> {
    pub(crate) world: &'a UserCommunicator,
    pub(crate) ns_p: *mut petsc_raw::_p_MatNullSpace,
}

impl Drop for NullSpace<'_> {
    fn drop(&mut self) {
        let ierr = unsafe { petsc_raw::MatNullSpaceDestroy(&mut self.ns_p as *mut _) };
        let _ = Petsc::check_error(self.world, ierr); // TODO: should I unwrap or what idk?
    }
}

pub use petsc_raw::MatAssemblyType;
pub use petsc_raw::MatOption;
pub use petsc_raw::MatDuplicateOption;
pub use petsc_raw::MatStencil;
pub use petsc_raw::MatOperation;
use petsc_raw::MatReuse;

/// [`Mat`] Type
pub type MatType = crate::petsc_raw::MatTypeEnum;

/// Specifies a matrix operation that has a "`Mat` `Vector` `Vector`" function signature.
///
/// You would use [`Mat::shell_set_operation_mvv()`] with a closure that has the following
/// signature `FnMut(&Mat, &Vector, &mut Vector) -> Result<()>`.
///
/// This implements [`From`] and [`Into`] with [`MatOperation`] so you don't have to use
/// this enum directly.
// Note, the C API specifically defines the operations with numbers so
// it should be fine to also rely on that here.
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum MatOperationMVV {
    /// op for [`Mat::mult()`]
    MATOP_MULT = 3,
    /// op for [`Mat::mult_transpose()`]
    MATOP_MULT_TRANSPOSE = 5,
    // TODO: Does solve allow for the b vec to be NULL? if so then this should be here
    /// op for `MatSolve()`
    MATOP_SOLVE = 7,
    /// op for `MatSolveTranspose()`
    MATOP_SOLVE_TRANSPOSE = 9,
    // There are probably more that have the correct function signature that can be
    // added in the future. If you add any entries here, you must also add them to the
    // `impl From<MatOperation> for MatOperationMVV` at the bottom of the file and to the
    // table bellow. You also need to change the size use by the seq! macro in
    // `shell_set_operation_mvv` (there is a comment there).
}

// Note, this is usize because it is used for indexing. It doesn't matter what
// repr type MatOperationMVV uses.
static MAT_OPERATION_MVV_TABLE: [usize; 4] = [3,5,7,9];

/// Specifies a matrix operation that has a "`Mat` `Vector` `Vector` `Vector`" function signature.
///
/// You would use [`Mat::shell_set_operation_mvvv()`] with a closure that has the following
/// signature `FnMut(&Mat, &Vector, &Vector, &mut Vector) -> Result<()>`.
///
/// This implements [`From`] and [`Into`] with [`MatOperation`] so you don't have to use
/// this enum directly.
// Note, the C API specifically defines the operations with numbers so
// it should be fine to also rely on that here.
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum MatOperationMVVV {
    /// op for [`Mat::mult_add()`]
    MATOP_MULT_ADD = 5,
    /// op for [`Mat::mult_transpose_add()`]
    MATOP_MULT_TRANSPOSE_ADD = 6,
    /// op for `MatSolveAdd()`
    MATOP_SOLVE_ADD = 8,
    /// op for `MatSolveTransposeAdd()`
    MATOP_SOLVE_TRANSPOSE_ADD = 10,
    // There are probably more that have the correct function signature that can be
    // added in the future. If you add any entries here, you must also add them to the
    // `impl From<MatOperation> for MatOperationMVVV` at the bottom of the file and to the
    // table bellow. You also need to change the size use by the seq! macro in
    // `shell_set_operation_mvvv` (there is a comment there).
}

static MAT_OPERATION_MVVV_TABLE: [usize; 4] = [4,6,8,10];

/// Specifies a matrix operation that has a "`Mat` `Vector`" function signature.
///
/// You would use [`Mat::shell_set_operation_mv()`] with a closure that has the following
/// signature `FnMut(&Mat, &mut Vector) -> Result<()>`.
///
/// This implements [`From`] and [`Into`] with [`MatOperation`] so you don't have to use
/// this enum directly.
// Note, the C API specifically defines the operations with numbers so
// it should be fine to also rely on that here.
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum MatOperationMV {
    /// op for [`Mat::get_diagonal()`]
    MATOP_GET_DIAGONAL = 17,
    // There are probably more that have the correct function signature that can be
    // added in the future. If you add any entries here, you must also add them to the
    // `impl From<MatOperation> for MatOperationMV` at the bottom of the file and to the
    // table bellow. You also need to change the size use by the seq! macro in
    // `shell_set_operation_mv` (there is a comment there).
}

static MAT_OPERATION_MV_TABLE: [usize; 1] = [17];

enum MatShellSingleOperationTrampolineData<'a, 'tl> {
    MVVV(Box<dyn FnMut(&Mat<'a, 'tl>, &Vector<'a>, &Vector<'a>, &mut Vector<'a>) -> Result<()> + 'tl>),
    MVV(Box<dyn FnMut(&Mat<'a, 'tl>, &Vector<'a>, &mut Vector<'a>) -> Result<()> + 'tl>),
    MV(Box<dyn FnMut(&Mat<'a, 'tl>, &mut Vector<'a>) -> Result<()> + 'tl>),
}

struct MatShellTrampolineData<'a, 'tl> {
    #[allow(dead_code)]
    world: &'a UserCommunicator,
    // TODO: there are 148 ops, but this might change so we should get this number in a better way
    // Also if this number changes, this is not the only occurrence of it. You
    // will have to change it in other places too.
    user_funcs: [Option<MatShellSingleOperationTrampolineData<'a, 'tl>>; 148],
}

impl<'a, 'tl> Mat<'a, 'tl> {
    /// Same as `Mat { ... }` but sets all optional params to `None`
    pub(crate) fn new(world: &'a UserCommunicator, mat_p: *mut petsc_raw::_p_Mat) -> Self {
        Mat { world, mat_p, shell_trampoline_data: None, }
    }

    /// Same at [`Petsc::mat_create()`].
    pub fn create(world: &'a UserCommunicator,) -> Result<Self> {
        let mut mat_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::MatCreate(world.as_raw(), mat_p.as_mut_ptr()) };
        Petsc::check_error(world, ierr)?;

        Ok(Mat::new(world, unsafe { mat_p.assume_init() }))
    }

    /// Creates a new matrix class for use with a user-defined private data storage format. 
    pub fn create_shell(world: &'a UserCommunicator, local_rows: impl Into<Option<PetscInt>>,
        local_cols: impl Into<Option<PetscInt>>, global_rows: impl Into<Option<PetscInt>>,
        global_cols: impl Into<Option<PetscInt>>) -> Result<Self>
    {
        let none_array = seq!(N in 0..148 { [ #( None, )* ]});
        let ctx = Box::pin(MatShellTrampolineData { 
            world: world, user_funcs: none_array, });
        let mut mat_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::MatCreateShell(world.as_raw(),
            local_rows.into().unwrap_or(petsc_raw::PETSC_DECIDE_INTEGER),
            local_cols.into().unwrap_or(petsc_raw::PETSC_DECIDE_INTEGER),
            global_rows.into().unwrap_or(petsc_raw::PETSC_DECIDE_INTEGER),
            global_cols.into().unwrap_or(petsc_raw::PETSC_DECIDE_INTEGER),
            std::mem::transmute(ctx.as_ref()),
            mat_p.as_mut_ptr()) };
        Petsc::check_error(world, ierr)?;

        Ok(Mat { world, mat_p: unsafe { mat_p.assume_init() }, shell_trampoline_data: Some(ctx)})
    }

    /// Duplicates a matrix including the non-zero structure.
    ///
    /// Note, [`Mat::clone()`] is the same as `x.duplicate(MatDuplicateOption::MAT_COPY_VALUES)`.
    ///
    /// See the manual page for [`MatDuplicateOption`](https://petsc.org/release/docs/manualpages/Mat/MatDuplicateOption.html#MatDuplicateOption) for an explanation of these options.
    pub fn duplicate(&self, op: MatDuplicateOption) -> Result<Self> {
        let mut mat2_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::MatDuplicate(self.mat_p, op, mat2_p.as_mut_ptr()) };
        Petsc::check_error(self.world, ierr)?;

        Ok(Mat::new(self.world, unsafe { mat2_p.assume_init() }))
    }

    /// Sets the local and global sizes, and checks to determine compatibility
    ///
    /// For rows and columns, local and global cannot be both None. If one processor calls this with a global of None then all processors must, otherwise the program will hang.
    /// If None is not used for the local sizes, then the user must ensure that they are chosen to be compatible with the vectors.
    pub fn set_sizes(&mut self, local_rows: impl Into<Option<PetscInt>>, local_cols: impl Into<Option<PetscInt>>, global_rows: impl Into<Option<PetscInt>>, global_cols: impl Into<Option<PetscInt>>) -> Result<()> {
        let ierr = unsafe { petsc_raw::MatSetSizes(
            self.mat_p, local_rows.into().unwrap_or(petsc_raw::PETSC_DECIDE_INTEGER), 
            local_cols.into().unwrap_or(petsc_raw::PETSC_DECIDE_INTEGER), 
            global_rows.into().unwrap_or(petsc_raw::PETSC_DECIDE_INTEGER), 
            global_cols.into().unwrap_or(petsc_raw::PETSC_DECIDE_INTEGER)) };
        Petsc::check_error(self.world, ierr)
    }

    /// Inserts or adds a block of values into a matrix. 
    ///
    /// These values may be cached, so [`Mat::assembly_begin()`] and [`Mat::assembly_end()`] MUST 
    /// be called after all calls to [`Mat::set_values()`] have been completed.
    /// For more info read: <https://petsc.org/release/docs/manualpages/Mat/MatSetValues.html>
    ///
    /// If you create the matrix yourself (that is not with a call to DMCreateMatrix()) then you 
    /// MUST call some `set_preallocation()` (such as [`Mat::seq_aij_set_preallocation()`]) or 
    /// [`Mat::set_up`] before using this routine.
    ///
    /// Negative indices may be passed in idxm and idxn, these rows and columns are simply ignored.
    /// This allows easily inserting element stiffness matrices with homogeneous Dirchlet boundary
    /// conditions that you don't want represented in the matrix.
    ///
    /// You might find [`Mat::assemble_with()`] or [`Mat::assemble_with_batched()`] to be more useful and
    /// more idiomatic.
    ///
    /// # Parameters
    ///
    /// * `idxm` - the row indices to add values to
    /// * `idxn` - the column indices to add values to
    /// * `v` - a logivally two-dimensional array of values (of size `idxm.len() * idxn.len()`)
    /// * `addv` - Either [`INSERT_VALUES`](InsertMode::INSERT_VALUES) or [`ADD_VALUES`](InsertMode::ADD_VALUES), 
    /// where [`ADD_VALUES`](InsertMode::ADD_VALUES) adds values to any existing entries, and 
    /// [`INSERT_VALUES`](InsertMode::INSERT_VALUES) replaces existing entries with new values.
    ///
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # use mpi::traits::*;
    /// # fn main() -> petsc_rs::Result<()> {
    /// # let petsc = Petsc::init_no_args().unwrap();
    /// if petsc.world().size() != 1 {
    ///     // note, cargo wont run tests with mpi so this will never be reached,
    ///     // but this example will only work in a uniprocessor comm world
    ///     Petsc::set_error(petsc.world(), PetscErrorKind::PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!")?;
    /// }
    ///
    /// let n = 3;
    /// let mut mat = petsc.mat_create()?;
    /// mat.set_sizes(None, None, Some(n), Some(n))?;
    /// mat.set_from_options()?;
    /// mat.set_up()?;
    /// let mut mat2 = mat.duplicate(MatDuplicateOption::MAT_DO_NOT_COPY_VALUES)?;
    ///
    /// // We will create two matrices that look like the following:
    /// //  0  1  2
    /// //  3  4  5
    /// //  6  7  8
    /// for i in 0..n {
    ///     let v = [PetscScalar::from(i as PetscReal) * 3.0, PetscScalar::from(i as PetscReal) * 3.0+1.0,
    ///         PetscScalar::from(i as PetscReal) * 3.0+2.0];
    ///     mat.set_values(&[i], &[0,1,2], &v, InsertMode::INSERT_VALUES)?;
    /// }
    /// // You MUST assemble before you can use 
    /// mat.assembly_begin(MatAssemblyType::MAT_FINAL_ASSEMBLY)?;
    /// mat.assembly_end(MatAssemblyType::MAT_FINAL_ASSEMBLY)?;
    /// # // for debugging
    /// # let viewer = Viewer::create_ascii_stdout(petsc.world())?;
    /// # mat.view_with(Some(&viewer))?;
    ///
    /// for i in 0..n {
    ///     let v = [PetscScalar::from(i as PetscReal), PetscScalar::from(i as PetscReal) + 3.0,
    ///         PetscScalar::from(i as PetscReal) + 6.0];
    ///     mat2.set_values(&[0,1,2], &[i], &v, InsertMode::INSERT_VALUES)?;
    /// }
    /// // You MUST assemble before you can use 
    /// mat2.assembly_begin(MatAssemblyType::MAT_FINAL_ASSEMBLY)?;
    /// mat2.assembly_end(MatAssemblyType::MAT_FINAL_ASSEMBLY)?;
    /// # // for debugging
    /// # mat2.view_with(Some(&viewer))?;
    /// 
    /// // We do that map in the case that `PetscScalar` is complex.
    /// assert_eq!(mat.get_values(0..n, 0..n).unwrap(), mat2.get_values(0..n, 0..n).unwrap());
    /// assert_eq!(mat.get_values(0..n, 0..n).unwrap(), [ 0.0,  1.0,  2.0,
    ///                                                   3.0,  4.0,  5.0,
    ///                                                   6.0,  7.0,  8.0,]
    ///     .iter().cloned().map(|v| PetscScalar::from(v)).collect::<Vec<_>>());
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_values(&mut self, idxm: &[PetscInt], idxn: &[PetscInt], v: &[PetscScalar], addv: InsertMode) -> Result<()> {
        let m = idxm.len();
        let n = idxn.len();
        assert_eq!(v.len(), m*n);
        let ierr = unsafe { petsc_raw::MatSetValues(self.mat_p, m as PetscInt, idxm.as_ptr(), n as PetscInt,
            idxn.as_ptr(), v.as_ptr() as *mut _, addv) };
        Petsc::check_error(self.world, ierr)
    }

    /// Inserts or adds a block of values into a matrix. Using structured grid indexing
    ///
    /// The grid coordinates are across the entire grid, not just the local portion.
    ///
    /// In order to use this routine, the matrix must have been created by a [DM](crate::dm).
    ///
    /// The columns and rows in the stencil passed in MUST be contained within the ghost region of
    /// the given process as set with `DM::da_create_XXX()` or `MatSetStencil()`. For example, if you
    /// create a [`DMDA`](crate::dm) with an overlap of one grid level and on a particular process its
    /// first local nonghost x logical coordinate is 6 (so its first ghost x logical coordinate is 5)
    /// the first i index you can use in your column and row indices in `MatSetStencil()` is 5.
    ///
    /// C API docs: <https://petsc.org/release/docs/manualpages/Mat/MatSetValuesStencil.html#MatSetValuesStencil>
    ///
    /// # Notes for [`MatStencil`] type
    ///
    /// The `i`,`j`, and `k` represent the logical coordinates over the entire grid (for 2 and
    /// 1 dimensional problems the `k` and `j` entries are ignored). The `c` represents the the degrees
    /// of freedom at each grid point (the dof argument to DMDASetDOF()). If dof is 1 then this entry
    /// is ignored.
    pub fn set_values_stencil(&mut self, idxm: &[MatStencil], idxn: &[MatStencil], v: &[PetscScalar], addv: InsertMode) -> Result<()> {
        let m = idxm.len();
        let n = idxn.len();
        assert_eq!(v.len(), m*n);
        let ierr = unsafe { petsc_raw::MatSetValuesStencil(self.mat_p, m as PetscInt, idxm.as_ptr(), n as PetscInt,
            idxn.as_ptr(), v.as_ptr() as *mut _, addv) };
        Petsc::check_error(self.world, ierr)
    }

    /// Inserts or adds values into certain locations of a matrix, using a local numbering of the nodes.
    ///
    /// Similar to [`Mat::set_values()`].
    pub fn set_local_values(&mut self, idxm: &[PetscInt], idxn: &[PetscInt], v: &[PetscScalar], addv: InsertMode) -> Result<()> {
        let m = idxm.len();
        let n = idxn.len();
        assert_eq!(v.len(), m*n);
        let ierr = unsafe { petsc_raw::MatSetValuesLocal(self.mat_p, m as PetscInt, idxm.as_ptr(), n as PetscInt,
            idxn.as_ptr(), v.as_ptr() as *mut _, addv) };
        Petsc::check_error(self.world, ierr)
    }

    /// Returns the range of matrix rows owned by this processor.
    ///
    /// We assume that the matrix is laid out with the first n1 rows on the first processor,
    /// the next n2 rows on the second, etc. For certain parallel layouts this range may not
    /// be well defined.
    pub fn get_ownership_range(&self) -> Result<std::ops::Range<PetscInt>> {
        let mut low = MaybeUninit::<PetscInt>::uninit();
        let mut high = MaybeUninit::<PetscInt>::uninit();
        let ierr = unsafe { petsc_raw::MatGetOwnershipRange(self.mat_p, low.as_mut_ptr(), high.as_mut_ptr()) };
        Petsc::check_error(self.world, ierr)?;

        Ok(unsafe { low.assume_init()..high.assume_init() })
    }

    /// Returns the range of matrix rows owned by EACH processor.
    ///
    /// We assume that the matrix is laid out with the first n1 rows on the first processor,
    /// the next n2 rows on the second, etc. For certain parallel layouts this range may not
    /// be well defined.
    pub fn get_ownership_ranges(&self) -> Result<Vec<std::ops::Range<PetscInt>>> {
        let mut array = MaybeUninit::<*const PetscInt>::uninit();
        let ierr = unsafe { petsc_raw::MatGetOwnershipRanges(self.mat_p, array.as_mut_ptr()) };
        Petsc::check_error(self.world, ierr)?;

        // SAFETY: Petsc says it is an array of length size+1
        let slice_from_array = unsafe { 
            std::slice::from_raw_parts(array.assume_init(), self.world.size() as usize + 1) };
        let array_iter = slice_from_array.iter();
        let mut slice_iter_p1 = slice_from_array.iter();
        let _ = slice_iter_p1.next();
        Ok(array_iter.zip(slice_iter_p1).map(|(s,e)| *s..*e).collect())
    }

    /// Allows you to give an iter that will be use to make a series of calls to [`Mat::set_values`].
    /// Then is followed by both [`Mat::assembly_begin()`] and [`Mat::assembly_end()`].
    ///
    /// Note, each call to [`Mat::set_values()`] will only add only one value to the matrix at
    /// a time. If you want to insert multiple at a time, use [`Mat::assemble_with_batched()`].
    ///
    /// [`assemble_with()`](Mat::assemble_with()) will short circuit on the first error
    /// from [`Mat::set_values`], returning the `Err`.
    ///
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # use mpi::traits::*;
    /// # fn main() -> petsc_rs::Result<()> {
    /// # let petsc = Petsc::init_no_args().unwrap();
    /// if petsc.world().size() != 1 {
    ///     // note, cargo wont run tests with mpi so this will never be reached,
    ///     // but this example will only work in a uniprocessor comm world
    ///     Petsc::set_error(petsc.world(), PetscErrorKind::PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!")?;
    /// }
    ///
    /// let n = 5;
    /// let mut mat = petsc.mat_create()?;
    /// mat.set_sizes(None, None, Some(n), Some(n))?;
    /// mat.set_from_options()?;
    /// mat.set_up()?;
    ///
    /// // We will create a matrix that look like the following:
    /// //  2 -1  0  0  0
    /// // -1  2 -1  0  0
    /// //  0 -1  2 -1  0
    /// //  0  0 -1  2 -1
    /// //  0  0  0 -1  2
    /// mat.assemble_with((0..n).map(|i| (-1..=1).map(move |j| (i,i+j))).flatten()
    ///         .filter(|&(i, j)| j < n) // we could also filter out negatives, but `Mat::set_values()` will do that for us
    ///         .map(|(i,j)| if i == j { (i, j, PetscScalar::from(2.0)) }
    ///                      else { (i, j, PetscScalar::from(-1.0)) }),
    ///     InsertMode::INSERT_VALUES, MatAssemblyType::MAT_FINAL_ASSEMBLY)?;
    /// # // for debugging
    /// # let viewer = Viewer::create_ascii_stdout(petsc.world())?;
    /// # mat.view_with(Some(&viewer))?;
    /// 
    /// // We do that map in the case that `PetscScalar` is complex.
    /// assert_eq!(mat.get_values(0..n, 0..n).unwrap(), [ 2.0, -1.0,  0.0,  0.0,  0.0,
    ///                                                  -1.0,  2.0, -1.0,  0.0,  0.0,
    ///                                                   0.0, -1.0,  2.0, -1.0,  0.0,
    ///                                                   0.0,  0.0, -1.0,  2.0, -1.0,
    ///                                                   0.0,  0.0,  0.0, -1.0,  2.0]
    ///     .iter().cloned().map(|v| PetscScalar::from(v)).collect::<Vec<_>>());
    /// # Ok(())
    /// # }
    /// ```
    pub fn assemble_with<I>(&mut self, iter_builder: I, addv: InsertMode, assembly_type: MatAssemblyType) -> Result<()>
    where
        I: IntoIterator<Item = (PetscInt, PetscInt, PetscScalar)>
    {
        // We don't actually care about the num_inserts value, we just need something that
        // implements `Sum` so we can use the sum method and `()` does not.
        let _num_inserts = iter_builder.into_iter().map(|(idxm, idxn, v)| {
            self.set_values(std::slice::from_ref(&idxm), std::slice::from_ref(&idxn),
                std::slice::from_ref(&v), addv).map(|_| 1)
        }).sum::<Result<PetscInt>>()?;
        // Note, `sum()` will short-circuit the iterator if an error is encountered.

        self.assembly_begin(assembly_type)?;
        self.assembly_end(assembly_type)
    }

    /// Allows you to give an iter that will be use to make a series of calls to [`Mat::set_values`].
    /// Then is followed by both [`Mat::assembly_begin()`] and [`Mat::assembly_end()`].
    ///
    /// Unlike [`assemble_with()`](Mat::assemble_with()), this will set values in batches, i.e., the
    /// input is given with array like values. They can be [`array`]s, [`slice`]s, [`Vec`]s, or anything that
    /// implements [`AsRef<[_]>`](AsRef).
    ///
    /// [`assemble_with_batched()`](Mat::assemble_with_batched()) will short circuit on the first error
    /// from [`Mat::set_values`], returning the `Err`.
    ///
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # use mpi::traits::*;
    /// # fn main() -> petsc_rs::Result<()> {
    /// # let petsc = Petsc::init_no_args().unwrap();
    /// if petsc.world().size() != 1 {
    ///     // note, cargo wont run tests with mpi so this will never be reached,
    ///     // but this example will only work in a uniprocessor comm world
    ///     Petsc::set_error(petsc.world(), PetscErrorKind::PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!")?;
    /// }
    ///
    /// let n = 5;
    /// let mut mat = petsc.mat_create()?;
    /// mat.set_sizes(None, None, Some(n), Some(n))?;
    /// mat.set_from_options()?;
    /// mat.set_up()?;
    ///
    /// // We will create a matrix that look like the following:
    /// //  2 -1  0  0  0
    /// // -1  2 -1  0  0
    /// //  0 -1  2 -1  0
    /// //  0  0 -1  2 -1
    /// //  0  0  0 -1  2
    /// let v = [PetscScalar::from(-1.0), PetscScalar::from(2.0), PetscScalar::from(-1.0)];
    /// mat.assemble_with_batched((0..n)
    ///         .map(|i| if i == 0 { ( vec![i], vec![i, i+1], &v[1..] ) }
    ///                  else if i == n-1 { ( vec![i], vec![i-1, i], &v[..2] ) }
    ///                  else { ( vec![i], vec![i-1, i, i+1], &v[..] ) }),
    ///     InsertMode::INSERT_VALUES, MatAssemblyType::MAT_FINAL_ASSEMBLY)?;
    /// # // for debugging
    /// # let viewer = Viewer::create_ascii_stdout(petsc.world())?;
    /// # mat.view_with(Some(&viewer))?;
    /// 
    /// assert_eq!(mat.get_values(0..n, 0..n).unwrap(), [ 2.0, -1.0,  0.0,  0.0,  0.0,
    ///                                                  -1.0,  2.0, -1.0,  0.0,  0.0,
    ///                                                   0.0, -1.0,  2.0, -1.0,  0.0,
    ///                                                   0.0,  0.0, -1.0,  2.0, -1.0,
    ///                                                   0.0,  0.0,  0.0, -1.0,  2.0]
    ///     .iter().cloned().map(|v| PetscScalar::from(v)).collect::<Vec<_>>());
    /// # Ok(())
    /// # }
    /// ```
    pub fn assemble_with_batched<I, A1, A2, A3>(&mut self, iter_builder: I, addv: InsertMode, assembly_type: MatAssemblyType) -> Result<()>
    where
        I: IntoIterator<Item = (A1, A2, A3)>,
        A1: AsRef<[PetscInt]>,
        A2: AsRef<[PetscInt]>,
        A3: AsRef<[PetscScalar]>,
    {
        // We don't actually care about the num_inserts value, we just need something that
        // implements `Sum` so we can use the sum method and `()` does not.
        let _num_inserts = iter_builder.into_iter().map(|(idxm, idxn, v)| {
            self.set_values(idxm.as_ref(), idxn.as_ref(), v.as_ref(), addv).map(|_| 1)
        }).sum::<Result<i32>>()?;
        // Note, `sum()` will short-circuit the iterator if an error is encountered.
        
        self.assembly_begin(assembly_type)?;
        self.assembly_end(assembly_type)
    }

    /// Allows you to give an iter that will be use to make a series of calls to [`Mat::set_values_stencil()`].
    /// Then is followed by both [`Mat::assembly_begin()`] and [`Mat::assembly_end()`].
    ///
    /// This functions identically to [`Mat::assemble_with()`] but uses [`Mat::set_values_stencil()`].
    pub fn assemble_with_stencil<I>(&mut self, iter_builder: I, addv: InsertMode, assembly_type: MatAssemblyType) -> Result<()>
    where
        I: IntoIterator<Item = (MatStencil, MatStencil, PetscScalar)>
    {
        // We don't actually care about the num_inserts value, we just need something that
        // implements `Sum` so we can use the sum method and `()` does not.
        let _num_inserts = iter_builder.into_iter().map(|(idxm, idxn, v)| {
            self.set_values_stencil(std::slice::from_ref(&idxm), std::slice::from_ref(&idxn),
                std::slice::from_ref(&v), addv).map(|_| 1)
        }).sum::<Result<PetscInt>>()?;
        // Note, `sum()` will short-circuit the iterator if an error is encountered.

        self.assembly_begin(assembly_type)?;
        self.assembly_end(assembly_type)
    }

    /// Allows you to give an iter that will be use to make a series of calls to [`Mat::set_values_stencil()`].
    /// Then is followed by both [`Mat::assembly_begin()`] and [`Mat::assembly_end()`].
    ///
    /// This functions identically to [`Mat::assemble_with_batched()`] but uses [`Mat::set_values_stencil()`].
    pub fn assemble_with_stencil_batched<I, A1, A2, A3>(&mut self, iter_builder: I, addv: InsertMode, assembly_type: MatAssemblyType) -> Result<()>
    where
        I: IntoIterator<Item = (A1, A2, A3)>,
        A1: AsRef<[MatStencil]>,
        A2: AsRef<[MatStencil]>,
        A3: AsRef<[PetscScalar]>,
    {
        // We don't actually care about the num_inserts value, we just need something that
        // implements `Sum` so we can use the sum method and `()` does not.
        let _num_inserts = iter_builder.into_iter().map(|(idxm, idxn, v)| {
            self.set_values_stencil(idxm.as_ref(), idxn.as_ref(), v.as_ref(), addv).map(|_| 1)
        }).sum::<Result<i32>>()?;
        // Note, `sum()` will short-circuit the iterator if an error is encountered.

        self.assembly_begin(assembly_type)?;
        self.assembly_end(assembly_type)
    }

    /// Allows you to give an iter that will be use to make a series of calls to [`Mat::set_local_values`].
    /// 
    /// Note, this is NOT followed by both [`Mat::assembly_begin()`] or [`Mat::assembly_end()`].
    /// You are required to make those calls.
    /// 
    /// Similar to [`Mat::assemble_with()`].
    pub fn set_local_values_with<I>(&mut self, iter_builder: I, addv: InsertMode) -> Result<()>
    where
        I: IntoIterator<Item = (PetscInt, PetscInt, PetscScalar)>
    {
        // We don't actually care about the num_inserts value, we just need something that
        // implements `Sum` so we can use the sum method and `()` does not.
        let _num_inserts = iter_builder.into_iter().map(|(idxm, idxn, v)| {
            self.set_local_values(std::slice::from_ref(&idxm), std::slice::from_ref(&idxn),
                std::slice::from_ref(&v), addv).map(|_| 1)
        }).sum::<Result<PetscInt>>()?;
        // Note, `sum()` will short-circuit the iterator if an error is encountered.
        Ok(())
    }

    /// Allows you to give an iter that will be use to make a series of calls to [`Mat::set_local_values`].
    /// 
    /// Note, this is NOT followed by both [`Mat::assembly_begin()`] or [`Mat::assembly_end()`].
    /// You are required to make those calls.
    /// 
    /// Similar to [`Mat::assemble_with_batched()`].
    pub fn set_local_values_with_batched<I, A1, A2, A3>(&mut self, iter_builder: I, addv: InsertMode) -> Result<()>
    where
        I: IntoIterator<Item = (A1, A2, A3)>,
        A1: AsRef<[PetscInt]>,
        A2: AsRef<[PetscInt]>,
        A3: AsRef<[PetscScalar]>,
    {
        // We don't actually care about the num_inserts value, we just need something that
        // implements `Sum` so we can use the sum method and `()` does not.
        let _num_inserts = iter_builder.into_iter().map(|(idxm, idxn, v)| {
            self.set_local_values(idxm.as_ref(), idxn.as_ref(), v.as_ref(), addv).map(|_| 1)
        }).sum::<Result<i32>>()?;
        // Note, `sum()` will short-circuit the iterator if an error is encountered.
        Ok(())
    }

    /// Gets values from certain locations of a Matrix. Currently can only get values on the same processor.
    ///
    /// # Example
    /// 
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # use mpi::traits::*;
    /// # fn main() -> petsc_rs::Result<()> {
    /// # let petsc = Petsc::init_no_args().unwrap();
    /// if petsc.world().size() != 1 {
    ///     // note, cargo wont run tests with mpi so this will never be reached,
    ///     // but this example will only work in a uniprocessor comm world
    ///     Petsc::set_error(petsc.world(), PetscErrorKind::PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!")?;
    /// }
    ///
    /// let n = 3;
    /// let mut mat = petsc.mat_create()?;
    /// mat.set_sizes(None, None, Some(n), Some(n))?;
    /// mat.set_from_options()?;
    /// mat.set_up()?;
    ///
    /// // We will create a matrix that look like the following:
    /// //  0  1  2
    /// //  3  4  5
    /// //  6  7  8
    /// mat.assemble_with((0..n).map(|i| (0..n).map(move |j| (i,j))).flatten().enumerate()
    ///         .map(|(v, (i,j))| (i,j,PetscScalar::from(v as PetscReal))),
    ///     InsertMode::INSERT_VALUES, MatAssemblyType::MAT_FINAL_ASSEMBLY)?;
    /// # // for debugging
    /// # let viewer = Viewer::create_ascii_stdout(petsc.world())?;
    /// # mat.view_with(Some(&viewer))?;
    /// 
    /// // We do that map in the case that `PetscScalar` is complex.
    /// assert_eq!(mat.get_values(0..n, 0..n).unwrap(),
    ///            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].iter().cloned().map(|v| PetscScalar::from(v)).collect::<Vec<_>>());
    /// assert_eq!(mat.get_values(0..n, [0]).unwrap(),
    ///            [0.0, 3.0, 6.0].iter().cloned().map(|v| PetscScalar::from(v)).collect::<Vec<_>>());
    /// assert_eq!(mat.get_values([1], 0..2).unwrap(),
    ///            [3.0, 4.0].iter().cloned().map(|v| PetscScalar::from(v)).collect::<Vec<_>>());
    /// assert_eq!(mat.get_values([1,2], [0,2]).unwrap(),
    ///            [3.0, 5.0, 6.0, 8.0].iter().cloned().map(|v| PetscScalar::from(v)).collect::<Vec<_>>());
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_values<T1, T2>(&self, idxm: T1, idxn: T2) -> Result<Vec<PetscScalar>>
    where
        T1: IntoIterator<Item = PetscInt>,
        T2: IntoIterator<Item = PetscInt>,
    {
        // TODO: make this return an ndarray
        let idxm_iter = idxm.into_iter();
        let idxm_array = idxm_iter.collect::<Vec<_>>();
        let mi = idxm_array.len();

        let idxn_iter = idxn.into_iter();
        let idxn_array = idxn_iter.collect::<Vec<_>>();
        let ni = idxn_array.len();
        
        let mut out_vec = vec![PetscScalar::default(); mi * ni];

        let ierr = unsafe { petsc_raw::MatGetValues(self.mat_p, mi as PetscInt, idxm_array.as_ptr(),
            ni as PetscInt, idxn_array.as_ptr(), out_vec[..].as_mut_ptr() as *mut _) };
        Petsc::check_error(self.world, ierr)?;

        Ok(out_vec)
    }

    ///  Assembles the matrix by calling [`Mat::assembly_begin()`] then [`Mat::assembly_end()`]
    pub fn assemble(&mut self, assembly_type: MatAssemblyType) -> Result<()>
    {
        self.assembly_begin(assembly_type)?;
        // TODO: what would even go here?
        self.assembly_end(assembly_type)
    }

    /// Performs Matrix-Matrix Multiplication `C=self*other`.
    ///
    /// Expected fill as ratio of `nnz(C)/(nnz(self) + nnz(other))`, use `None` if you do not have
    /// a good estimate. If the result is a dense matrix this is irrelevant.
    pub fn mat_mult(&self, other: &Mat, fill: impl Into<Option<PetscReal>>) -> Result<Self>{
        let mut mat_out_p = MaybeUninit::uninit();
        // TODO: do we want other MatReuse options
        let ierr = unsafe { petsc_raw::MatMatMult(self.mat_p, other.mat_p, MatReuse::MAT_INITIAL_MATRIX,
            fill.into().unwrap_or(petsc_raw::PETSC_DEFAULT_REAL), mat_out_p.as_mut_ptr()) };
        Petsc::check_error(self.world, ierr)?;

        Ok(Mat::new(self.world, unsafe { mat_out_p.assume_init() }))
    }

    /// Attaches a null space to a matrix.
    ///
    /// This null space is used by the linear solvers. Overwrites any previous null space that may
    /// have been attached. You can remove the null space by calling this routine with `None`.
    ///
    /// Krylov solvers can produce the minimal norm solution to the least squares problem by utilizing
    /// [`NullSpace::remove_from()`].
    // TODO: I don't like this api, we only force it to be a `Rc` because we dont want the caller editing
    // the nullspace after they set it (we don't actually use it to reference count).
    pub fn set_nullspace(&mut self, nullspace: impl Into<Option<Rc<NullSpace<'a>>>>) -> Result<()> {
        let ierr = unsafe { petsc_raw::MatSetNullSpace(self.mat_p,
            nullspace.into().as_ref().map_or(std::ptr::null_mut(), |ns| ns.ns_p)) };
        Petsc::check_error(self.world, ierr)
    }

    /// Attaches a left null space to a matrix.
    ///
    /// This null space is used by the linear solvers. Overwrites any previous null space that may
    /// have been attached. You can remove the null space by calling this routine with `None`.
    ///
    /// Krylov solvers can produce the minimal norm solution to the least squares problem by utilizing
    /// [`NullSpace::remove_from()`].
    pub fn set_left_nullspace(&mut self, nullspace: impl Into<Option<Rc<NullSpace<'a>>>>) -> Result<()> {
        let ierr = unsafe { petsc_raw::MatSetTransposeNullSpace(self.mat_p,
            nullspace.into().as_ref().map_or(std::ptr::null_mut(), |ns| ns.ns_p)) };
        Petsc::check_error(self.world, ierr)
    }

    /// Attaches a null space to a matrix, which is often the null space (rigid body modes)
    /// of the operator without boundary conditions This null space will be used to provide
    /// near null space vectors to a multigrid preconditioner built from this matrix.
    ///
    /// This null space is used by the linear solvers. Overwrites any previous null space that may
    /// have been attached. You can remove the null space by calling this routine with `None`.
    pub fn set_near_nullspace(&mut self, nullspace: impl Into<Option<Rc<NullSpace<'a>>>>) -> Result<()> {
        let ierr = unsafe { petsc_raw::MatSetNearNullSpace(self.mat_p,
            nullspace.into().as_ref().map_or(std::ptr::null_mut(), |ns| ns.ns_p)) };
        Petsc::check_error(self.world, ierr)
    }

    /// Gets a reference to a submatrix specified in local numbering.
    ///
    /// # Notes
    ///
    /// Depending on the format of mat, the returned submat may not implement [`Mat::mult()`]
    /// or other functions. Its communicator may be the same as self, it may be `PETSC_COMM_SELF`,
    /// or some other subcomm of mat's. 
    ///
    /// The submat always implements [`Mat::set_local_values()`] (and thus you can also use
    /// [`Mat::set_local_values_with()`]).
    pub fn get_local_sub_matrix_mut<'bv>(&'bv mut self, is_row: Rc<IS<'a>>, is_col: Rc<IS<'a>>) -> Result<BorrowMatMut<'a, 'tl, 'bv>> {
        // TODO: make a non-mut version of this function
        let mut mat_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::MatGetLocalSubMatrix(self.mat_p,
            is_row.is_p, is_col.is_p, mat_p.as_mut_ptr()) };
        Petsc::check_error(self.world, ierr)?;
        Ok(BorrowMatMut::new( 
            ManuallyDrop::new( Mat::new(self.world, unsafe { mat_p.assume_init() })),
            Some(Box::new(move |borrow_mat| {
                    let ierr = unsafe { petsc_raw::MatRestoreLocalSubMatrix(self.mat_p,
                        is_row.is_p, is_col.is_p, &mut borrow_mat.owned_mat.mat_p as *mut _) };
                    let _ = Petsc::check_error(self.world, ierr); // TODO: should I unwrap ?
                })),
        ))
    }

    

    /// Builds [`Mat`] for a particular type
    pub fn set_type_str(&mut self, mat_type: &str) -> Result<()> {
        let cstring = CString::new(mat_type).expect("`CString::new` failed");
        let ierr = unsafe { petsc_raw::MatSetType(self.mat_p, cstring.as_ptr()) };
        Petsc::check_error(self.world, ierr)
    }

    /// Builds [`Mat`] for a particular type
    pub fn set_type(&mut self, mat_type: MatType) -> Result<()> {
        let option_cstr = petsc_raw::MATTYPE_TABLE[mat_type as usize];
        let ierr = unsafe { petsc_raw::MatSetType(self.mat_p, option_cstr.as_ptr() as *const _) };
        Petsc::check_error(self.world, ierr)
    }

    /// Determines whether a PETSc [`Mat`] is of a particular type.
    pub fn type_compare(&self, type_kind: MatType) -> Result<bool> {
        self.type_compare_str(&type_kind.to_string())
    }

    // TODO: add set shell matrix data function that uses MatShellSetContext, maybe
    // This would allow for operations that mutate the mat, with this method alone
    // we would get a multiple mutable references error.
    // in order to do this, however, i think would would have to use a generic and 
    // a new MatShell type. For now it could deref into a Mat, but in the future it might
    // make sense for it to impl a Mat trait.

    // TODO: add support for more types of ops. There are two ways i can think of doing it:
    //     1. Make a different function for each type of method - this could be confusing to
    //        the user, i.e. knowing what is supported and where. Or to solve this we can make a
    //        different enum for each method type. This would also make the trampoline type easier.
    //        We could also implement Into into each of those types from the base type. idk.
    //        We could basically use the same strategy that we are now with the `seq!` macro.
    //     2. Make a new MatOperation enum that contains the rust closure type and have one 
    //        `shell_set_operation` function do all the work. This would mean that we would take Box<dyn _>
    //        and not a generic like we do now.
    // Both of these we could slowly roll out one function at a time. Also, it seems like this will
    // be very tedious either way

    /// Allows user to set a matrix operation for a shell matrix.
    ///
    /// You can only set operations that expect the correct function signature:
    /// `FnMut(&Mat, &Vector, &mut Vector) -> Result<()>`
    ///
    /// This function only works for operations in [`MatOperationMVV`].
    ///
    /// # Parameters
    ///
    /// * `op` - the name of the operation
    /// * `user_f` - the name of the operation
    ///     * `mat` - The matrix
    ///     * `x` - The input vector
    ///     * `y` *(output)* - The output vector
    ///
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # use mpi::traits::*;
    /// # use ndarray::{s, array};
    /// # fn main() -> petsc_rs::Result<()> {
    /// # let petsc = Petsc::init_no_args()?;
    /// // Note: this example will only work in a uniprocessor comm world. Also, right
    /// // now this example only works when `PetscScalar` is `PetscReal`. It will fail
    /// // to compile if `PetscScalar` is `PetscComplex`.
    /// let mut x = Vector::from_slice(petsc.world(), &[1.2, -0.5])?;
    /// let mut y = Vector::from_slice(petsc.world(), &[0.0, 0.0])?;
    ///
    /// let theta = std::f64::consts::PI as PetscReal / 2.0;
    /// let mat_data = [PetscScalar::cos(theta), -PetscScalar::sin(theta),
    ///                 PetscScalar::sin(theta),  PetscScalar::cos(theta)];
    /// let mut mat = Mat::create_shell(petsc.world(),2,2,2,2)?;
    /// mat.set_up()?;
    ///
    /// mat.shell_set_operation_mvv(MatOperation::MATOP_MULT, |_m, x, y| {
    ///     let xx = x.view()?;
    ///     let mut yy = y.view_mut()?;
    ///     yy[0] = mat_data[0] * xx[0] + mat_data[1] * xx[1];
    ///     yy[1] = mat_data[2] * xx[0] + mat_data[3] * xx[1];
    ///     Ok(())
    /// })?;
    ///
    /// mat.shell_set_operation_mvv(MatOperation::MATOP_MULT_TRANSPOSE, |_m, x, y| {
    ///     let xx = x.view()?;
    ///     let mut yy = y.view_mut()?;
    ///     yy[0] = mat_data[0] * xx[0] + mat_data[2] * xx[1];
    ///     yy[1] = mat_data[1] * xx[0] + mat_data[3] * xx[1];
    ///     Ok(())
    /// })?;
    ///
    /// mat.mult(&x, &mut y)?;
    /// assert!(y.view()?.slice(s![..]).abs_diff_eq(&array![0.5, 1.2], 1e-15));
    /// mat.mult_transpose(&y, &mut x)?;
    /// assert!(x.view()?.slice(s![..]).abs_diff_eq(&array![1.2, -0.5], 1e-15));
    /// # Ok(())
    /// # }
    /// ```
    pub fn shell_set_operation_mvv<F>(&mut self, op: impl Into<MatOperationMVV>, user_f: F) -> Result<()>
    where
        F: FnMut(&Mat<'a, 'tl>, &Vector<'a>, &mut Vector<'a>) -> Result<()> + 'tl
    {
        let op: MatOperationMVV = op.into();
        let closure_anchor = MatShellSingleOperationTrampolineData::MVV(Box::new(user_f));

        if let Some(td) = self.shell_trampoline_data.as_mut() {
            let _ = td.as_mut().user_funcs[op as usize].take();
            td.as_mut().user_funcs[op as usize] = Some(closure_anchor);
        } else {
            let none_array = seq!(N in 0..148 { [ #( None, )* ]});
            let mut td = MatShellTrampolineData { 
                world: self.world, user_funcs: none_array };
            td.user_funcs[op as usize] = Some(closure_anchor);
            let td_anchor = Box::pin(td);
            let ierr = unsafe { petsc_raw::MatShellSetContext(self.mat_p,
                std::mem::transmute(td_anchor.as_ref())) }; // this will also erase the lifetimes
            Petsc::check_error(self.world, ierr)?;
            self.shell_trampoline_data = Some(td_anchor);
        }

        // The `MatOperationMVV` enum has 4 variants so we want to create 4 functions.
        // We use the `MatOperationMVVTable` to get what the correct index is.
        // If you change `MatOperationMVV`, then you have to update the number 4 used by the seq!
        // macro bellow to be the number of variants in `MatOperationMVV`, or the number of elements
        // in `MatOperationMVVTable`. Sadly, this macro expects a int literal, so there is no easy way
        // to automatically update it using a const or another macro. There is also another usage of
        // seq! bellow that you have to update.
        seq!(N in 0..4 {
            debug_assert!(N < MAT_OPERATION_MVV_TABLE.len(),
                "Internal Error: `shell_set_operation_mvv` was not updated, but `MatOperationMVVTable` was.");
            unsafe extern "C" fn mat_shell_operation_mvv_trampoline_#N (mat_p: *mut petsc_raw::_p_Mat, x_p: *mut petsc_raw::_p_Vec,
                y_p: *mut petsc_raw::_p_Vec) -> petsc_raw::PetscErrorCode
            {
                let mut ctx = MaybeUninit::<*mut ::std::os::raw::c_void>::uninit();
                // TODO: why does this one take a void* but `PCShellGetContext` takes a void**?
                // I looks like under the hood it is treated like a void** so idk
                let ierr = petsc_raw::MatShellGetContext(mat_p, ctx.as_mut_ptr() as *mut _);
                assert_eq!(ierr, 0);

                // SAFETY: TODO
                let trampoline_data: Pin<&mut MatShellTrampolineData> = std::mem::transmute(ctx.assume_init());
                let world = trampoline_data.world;

                let mat = ManuallyDrop::new(Mat::new(world, mat_p));
                let x = ManuallyDrop::new(Vector {world, vec_p: x_p });
                let mut y = ManuallyDrop::new(Vector {world, vec_p: y_p });
                
                (trampoline_data.get_unchecked_mut().user_funcs[MAT_OPERATION_MVV_TABLE[N]].as_mut()
                    .map_or_else(
                        || Petsc::set_error(world, PetscErrorKind::PETSC_ERR_ARG_CORRUPT,
                            format!(
                                "Rust function for {:?} was not found",
                                std::mem::transmute::<u32, MatOperation>(MAT_OPERATION_MVV_TABLE[N] as u32))),
                        |f| if let MatShellSingleOperationTrampolineData::MVV(f) = f {
                                (*f)(&mat, &x, &mut y)
                            } else {
                                // This should never happen
                                Petsc::set_error(world, PetscErrorKind::PETSC_ERR_ARG_CORRUPT,
                                    format!("Rust closure for Mat Op {:?} is the wrong type",
                                    std::mem::transmute::<u32, MatOperation>(MAT_OPERATION_MVV_TABLE[N] as u32)))
                            } ))
                    .map_or_else(|err| err.kind as i32, |_| 0)
            }
        });
        // If you change `MatOperationMVV`, then you have to update the number 4 used by the seq!
        // macro bellow to be the number of variants in `MatOperationMVV`, or the number of elements
        // in `MatOperationMVVTable`.
        let mut trampolines = [mat_shell_operation_mvv_trampoline_0
            as unsafe extern "C" fn(_, _, _) -> _;148];
        seq!(N in 0..4 {
            debug_assert!(N < MAT_OPERATION_MVV_TABLE.len(),
                "Internal Error: `shell_set_operation_mvv` was not updated, but `MatOperationMVVTable` was.");
            trampolines[MAT_OPERATION_MVV_TABLE[N]] = mat_shell_operation_mvv_trampoline_#N;
        });

        let mat_shell_operation_trampoline_ptr: ::std::option::Option<
            unsafe extern "C" fn(mat_p: *mut petsc_raw::_p_Mat, x_p: *mut petsc_raw::_p_Vec,
            y_p: *mut petsc_raw::_p_Vec, ) -> petsc_raw::PetscErrorCode, >
            = Some(trampolines[op as usize]);

        let ierr = unsafe { petsc_raw::MatShellSetOperation(self.mat_p, op.into(),
            std::mem::transmute(mat_shell_operation_trampoline_ptr)) }; // this will also erase the lifetimes
        Petsc::check_error(self.world, ierr)?;

        Ok(())
    }

    /// Allows user to set a matrix operation for a shell matrix.
    ///
    /// Works in the same way [`Mat::shell_set_operation_mvv()`] works, but you can only set operations
    /// that expect the function signature:
    /// `FnMut(&Mat, &mut Vector) -> Result<()>`
    ///
    /// This function only works for operations in [`MatOperationMV`].
    ///
    /// # Parameters
    ///
    /// * `op` - the name of the operation
    /// * `user_f` - the name of the operation
    ///     * `mat` - The matrix
    ///     * `v` *(output)* - The output vector
    ///
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # use mpi::traits::*;
    /// # use ndarray::{s, array};
    /// # fn main() -> petsc_rs::Result<()> {
    /// # let petsc = Petsc::init_no_args()?;
    /// // Note: this example will only work in a uniprocessor comm world. Also, right
    /// // now this example only works when `PetscScalar` is `PetscReal`. It will fail
    /// // to compile if `PetscScalar` is `PetscComplex`.
    /// let mut v = Vector::from_slice(petsc.world(), &[0.0, 0.0])?;
    ///
    /// let theta = std::f64::consts::PI as PetscReal / 2.0;
    /// let mat_data = [PetscScalar::cos(theta), -PetscScalar::sin(theta),
    ///                 PetscScalar::sin(theta),  PetscScalar::cos(theta)];
    /// let mut mat = Mat::create_shell(petsc.world(),2,2,2,2)?;
    /// mat.set_up()?;
    ///
    /// mat.shell_set_operation_mv(MatOperation::MATOP_GET_DIAGONAL, |_m, v| {
    ///     let mut vv = v.view_mut()?;
    ///     vv[0] = mat_data[0];
    ///     vv[1] = mat_data[3];
    ///     Ok(())
    /// })?;
    ///
    /// mat.get_diagonal(&mut v)?;
    /// assert!(v.view()?.slice(s![..]).abs_diff_eq(&array![0.0, 0.0], 1e-15));
    /// # Ok(())
    /// # }
    /// ```
    pub fn shell_set_operation_mv<F>(&mut self, op: impl Into<MatOperationMV>, user_f: F) -> Result<()>
    where
        F: FnMut(&Mat<'a, 'tl>, &mut Vector<'a>) -> Result<()> + 'tl
    {
        let op: MatOperationMV = op.into();
        let closure_anchor = MatShellSingleOperationTrampolineData::MV(Box::new(user_f));

        if let Some(td) = self.shell_trampoline_data.as_mut() {
            let _ = td.as_mut().user_funcs[op as usize].take();
            td.as_mut().user_funcs[op as usize] = Some(closure_anchor);
        } else {
            let none_array = seq!(N in 0..148 { [ #( None, )* ]});
            let mut td = MatShellTrampolineData { 
                world: self.world, user_funcs: none_array };
            td.user_funcs[op as usize] = Some(closure_anchor);
            let td_anchor = Box::pin(td);
            let ierr = unsafe { petsc_raw::MatShellSetContext(self.mat_p,
                std::mem::transmute(td_anchor.as_ref())) }; // this will also erase the lifetimes
            Petsc::check_error(self.world, ierr)?;
            self.shell_trampoline_data = Some(td_anchor);
        }

        // The `MatOperationMV` enum has 1 variants so we want to create 1 functions.
        // We use the `MatOperationMVTable` to get what the correct index is.
        // If you change `MatOperationMV`, then you have to update the number 1 used by the seq!
        // macro bellow to be the number of variants in `MatOperationMV`, or the number of elements
        // in `MatOperationMVTable`. Sadly, this macro expects a int literal, so there is no easy way
        // to automatically update it using a const or another macro. There is also another usage of
        // seq! bellow that you have to update.
        seq!(N in 0..1 {
            debug_assert!(N < MAT_OPERATION_MV_TABLE.len(),
                "Internal Error: `shell_set_operation_mv` was not updated, but `MatOperationMVTable` was.");
            unsafe extern "C" fn mat_shell_operation_mv_trampoline_#N (mat_p: *mut petsc_raw::_p_Mat,
                v_p: *mut petsc_raw::_p_Vec) -> petsc_raw::PetscErrorCode
            {
                let mut ctx = MaybeUninit::<*mut ::std::os::raw::c_void>::uninit();
                // TODO: why does this one take a void* but `PCShellGetContext` takes a void**?
                // I looks like under the hood it is treated like a void** so idk
                let ierr = petsc_raw::MatShellGetContext(mat_p, ctx.as_mut_ptr() as *mut _);
                assert_eq!(ierr, 0);

                // SAFETY: TODO
                let trampoline_data: Pin<&mut MatShellTrampolineData> = std::mem::transmute(ctx.assume_init());
                let world = trampoline_data.world;

                let mat = ManuallyDrop::new(Mat::new(world, mat_p));
                let mut v = ManuallyDrop::new(Vector {world, vec_p: v_p });
                
                (trampoline_data.get_unchecked_mut().user_funcs[MAT_OPERATION_MV_TABLE[N]].as_mut()
                    .map_or_else(
                        || Petsc::set_error(world, PetscErrorKind::PETSC_ERR_ARG_CORRUPT,
                            format!(
                                "Rust function for {:?} was not found",
                                std::mem::transmute::<u32, MatOperation>(MAT_OPERATION_MV_TABLE[N] as u32))),
                        |f| if let MatShellSingleOperationTrampolineData::MV(f) = f {
                                (*f)(&mat, &mut v)
                            } else {
                                // This should never happen
                                Petsc::set_error(world, PetscErrorKind::PETSC_ERR_ARG_CORRUPT,
                                    format!("Rust closure for Mat Op {:?} is the wrong type",
                                    std::mem::transmute::<u32, MatOperation>(MAT_OPERATION_MV_TABLE[N] as u32)))
                            } ))
                    .map_or_else(|err| err.kind as i32, |_| 0)
            }
        });
        // If you change `MatOperationMV`, then you have to update the number 1 used by the seq!
        // macro bellow to be the number of variants in `MatOperationMV`, or the number of elements
        // in `MatOperationMVTable`.
        let mut trampolines = [mat_shell_operation_mv_trampoline_0
            as unsafe extern "C" fn(_, _) -> _;148];
        seq!(N in 0..1 {
            debug_assert!(N < MAT_OPERATION_MV_TABLE.len(),
                "Internal Error: `shell_set_operation_mv` was not updated, but `MatOperationMVTable` was.");
            trampolines[MAT_OPERATION_MV_TABLE[N]] = mat_shell_operation_mv_trampoline_#N;
        });

        let mat_shell_operation_trampoline_ptr: ::std::option::Option<
            unsafe extern "C" fn(mat_p: *mut petsc_raw::_p_Mat,
            v_p: *mut petsc_raw::_p_Vec, ) -> petsc_raw::PetscErrorCode, >
            = Some(trampolines[op as usize]);

        let ierr = unsafe { petsc_raw::MatShellSetOperation(self.mat_p, op.into(),
            std::mem::transmute(mat_shell_operation_trampoline_ptr)) }; // this will also erase the lifetimes
        Petsc::check_error(self.world, ierr)?;

        Ok(())
    }

    /// Allows user to set a matrix operation for a shell matrix.
    ///
    /// Works in the same way [`Mat::shell_set_operation_mvv()`] works, but you can only set operations
    /// that expect the function signature:
    /// `FnMut(&Mat, &Vector, &Vector, &mut Vector) -> Result<()>`
    ///
    /// This function only works for operations in [`MatOperationMVVV`].
    ///
    /// # Parameters
    ///
    /// * `op` - the name of the operation
    /// * `user_f` - the name of the operation
    ///     * `mat` - The matrix
    ///     * `v1` - The first input vector
    ///     * `v2` - The second input vector
    ///     * `v3` *(output)* - The output vector
    ///
    pub fn shell_set_operation_mvvv<F>(&mut self, op: impl Into<MatOperationMVVV>, user_f: F) -> Result<()>
    where
        F: FnMut(&Mat<'a, 'tl>, &Vector<'a>, &Vector<'a>, &mut Vector<'a>) -> Result<()> + 'tl
    {
        let op: MatOperationMVVV = op.into();
        let closure_anchor = MatShellSingleOperationTrampolineData::MVVV(Box::new(user_f));

        if let Some(td) = self.shell_trampoline_data.as_mut() {
            let _ = td.as_mut().user_funcs[op as usize].take();
            td.as_mut().user_funcs[op as usize] = Some(closure_anchor);
        } else {
            let none_array = seq!(N in 0..148 { [ #( None, )* ]});
            let mut td = MatShellTrampolineData { 
                world: self.world, user_funcs: none_array };
            td.user_funcs[op as usize] = Some(closure_anchor);
            let td_anchor = Box::pin(td);
            let ierr = unsafe { petsc_raw::MatShellSetContext(self.mat_p,
                std::mem::transmute(td_anchor.as_ref())) }; // this will also erase the lifetimes
            Petsc::check_error(self.world, ierr)?;
            self.shell_trampoline_data = Some(td_anchor);
        }

        // The `MatOperationMVVV` enum has 4 variants so we want to create 4 functions.
        // We use the `MatOperationMVVVTable` to get what the correct index is.
        // If you change `MatOperationMVVV`, then you have to update the number 1 used by the seq!
        // macro bellow to be the number of variants in `MatOperationMVVV`, or the number of elements
        // in `MatOperationMVVVTable`. Sadly, this macro expects a int literal, so there is no easy way
        // to automatically update it using a const or another macro. There is also another usage of
        // seq! bellow that you have to update.
        seq!(N in 0..4 {
            debug_assert!(N < MAT_OPERATION_MV_TABLE.len(),
                "Internal Error: `shell_set_operation_mvvv` was not updated, but `MatOperationMVVVTable` was.");
            unsafe extern "C" fn mat_shell_operation_mvvv_trampoline_#N (mat_p: *mut petsc_raw::_p_Mat,
                v1_p: *mut petsc_raw::_p_Vec, v2_p: *mut petsc_raw::_p_Vec, v3_p: *mut petsc_raw::_p_Vec) -> petsc_raw::PetscErrorCode
            {
                let mut ctx = MaybeUninit::<*mut ::std::os::raw::c_void>::uninit();
                // TODO: why does this one take a void* but `PCShellGetContext` takes a void**?
                // I looks like under the hood it is treated like a void** so idk
                let ierr = petsc_raw::MatShellGetContext(mat_p, ctx.as_mut_ptr() as *mut _);
                assert_eq!(ierr, 0);

                // SAFETY: TODO
                let trampoline_data: Pin<&mut MatShellTrampolineData> = std::mem::transmute(ctx.assume_init());
                let world = trampoline_data.world;

                let mat = ManuallyDrop::new(Mat::new(world, mat_p));
                let v1 = ManuallyDrop::new(Vector {world, vec_p: v1_p });
                let v2 = ManuallyDrop::new(Vector {world, vec_p: v2_p });
                let mut v3 = ManuallyDrop::new(Vector {world, vec_p: v3_p });
                
                (trampoline_data.get_unchecked_mut().user_funcs[MAT_OPERATION_MVVV_TABLE[N]].as_mut()
                    .map_or_else(
                        || Petsc::set_error(world, PetscErrorKind::PETSC_ERR_ARG_CORRUPT,
                            format!(
                                "Rust function for {:?} was not found",
                                std::mem::transmute::<u32, MatOperation>(MAT_OPERATION_MVVV_TABLE[N] as u32))),
                        |f| if let MatShellSingleOperationTrampolineData::MVVV(f) = f {
                                (*f)(&mat, &v1, &v2, &mut v3)
                            } else {
                                // This should never happen
                                Petsc::set_error(world, PetscErrorKind::PETSC_ERR_ARG_CORRUPT,
                                    format!("Rust closure for Mat Op {:?} is the wrong type",
                                    std::mem::transmute::<u32, MatOperation>(MAT_OPERATION_MVVV_TABLE[N] as u32)))
                            } ))
                    .map_or_else(|err| err.kind as i32, |_| 0)
            }
        });
        // If you change `MatOperationMVVV`, then you have to update the number 1 used by the seq!
        // macro bellow to be the number of variants in `MatOperationMVVV`, or the number of elements
        // in `MatOperationMVVVTable`.
        let mut trampolines = [mat_shell_operation_mvvv_trampoline_0
            as unsafe extern "C" fn(_, _, _, _) -> _;148];
        seq!(N in 0..4 {
            debug_assert!(N < MAT_OPERATION_MV_TABLE.len(),
                "Internal Error: `shell_set_operation_mvvv` was not updated, but `MatOperationMVVVTable` was.");
            trampolines[MAT_OPERATION_MVVV_TABLE[N]] = mat_shell_operation_mvvv_trampoline_#N;
        });

        let mat_shell_operation_trampoline_ptr: ::std::option::Option<
            unsafe extern "C" fn(mat_p: *mut petsc_raw::_p_Mat, v1_p: *mut petsc_raw::_p_Vec,
                v2_p: *mut petsc_raw::_p_Vec, v3_p: *mut petsc_raw::_p_Vec, ) -> petsc_raw::PetscErrorCode, >
            = Some(trampolines[op as usize]);

        let ierr = unsafe { petsc_raw::MatShellSetOperation(self.mat_p, op.into(),
            std::mem::transmute(mat_shell_operation_trampoline_ptr)) }; // this will also erase the lifetimes
        Petsc::check_error(self.world, ierr)?;

        Ok(())
    }
}

impl Into<MatOperation> for MatOperationMVV {
    fn into(self) -> MatOperation {
        // Safety: The values of `MatOperationMVV` are always valid values of `MatOperation`
        // because we take them directly from `MatOperation`. Also, because the numeric values
        // of `MatOperation` are relied upon in the C API, it is safe to assume that as more
        // varients are added to `MatOperation`, none of the old ones will be touched.
        // Also the repr types for both enums are `u32` so memory layout/alignment will match.
        unsafe { std::mem::transmute(self) }
    }
}

impl From<MatOperation> for MatOperationMVV {
    /// This will panic if the value of `op` can't be a valid `MatOperationMVV`
    fn from(op: MatOperation) -> MatOperationMVV {
        match op {
            MatOperation::MATOP_MULT => MatOperationMVV::MATOP_MULT,
            MatOperation::MATOP_MULT_TRANSPOSE => MatOperationMVV::MATOP_MULT_TRANSPOSE,
            MatOperation::MATOP_SOLVE => MatOperationMVV::MATOP_SOLVE,
            MatOperation::MATOP_SOLVE_TRANSPOSE => MatOperationMVV::MATOP_SOLVE_TRANSPOSE,
            // There are more
            _ => panic!("The given op: `{:?}` can not be turned into a `MatOperationMVV`", op)
        }
    }
}

impl Into<MatOperation> for MatOperationMVVV {
    fn into(self) -> MatOperation {
        // Safety: The values of `MatOperationMVVV` are always valid values of `MatOperation`
        // because we take them directly from `MatOperation`. Also, because the numeric values
        // of `MatOperation` are relied upon in the C API, it is safe to assume that as more
        // varients are added to `MatOperation`, none of the old ones will be touched.
        // Also the repr types for both enums are `u32` so memory layout/alignment will match.
        unsafe { std::mem::transmute(self) }
    }
}

impl From<MatOperation> for MatOperationMVVV {
    /// This will panic if the value of `op` can't be a valid `MatOperationMVVV`
    fn from(op: MatOperation) -> MatOperationMVVV {
        match op {
            MatOperation::MATOP_MULT => MatOperationMVVV::MATOP_MULT_ADD,
            MatOperation::MATOP_MULT_TRANSPOSE => MatOperationMVVV::MATOP_MULT_TRANSPOSE_ADD,
            MatOperation::MATOP_SOLVE => MatOperationMVVV::MATOP_SOLVE_ADD,
            MatOperation::MATOP_SOLVE_TRANSPOSE => MatOperationMVVV::MATOP_SOLVE_TRANSPOSE_ADD,
            // There are more
            _ => panic!("The given op: `{:?}` can not be turned into a `MatOperationMVVV`", op)
        }
    }
}

impl Into<MatOperation> for MatOperationMV {
    fn into(self) -> MatOperation {
        // Safety: The values of `MatOperationMV` are always valid values of `MatOperation`
        // because we take them directly from `MatOperation`. Also, because the numeric values
        // of `MatOperation` are relied upon in the C API, it is safe to assume that as more
        // varients are added to `MatOperation`, none of the old ones will be touched.
        // Also the repr types for both enums are `u32` so memory layout/alignment will match.
        unsafe { std::mem::transmute(self) }
    }
}

impl From<MatOperation> for MatOperationMV {
    /// This will panic if the value of `op` can't be a valid `MatOperationMV`
    fn from(op: MatOperation) -> MatOperationMV {
        match op {
            MatOperation::MATOP_GET_DIAGONAL => MatOperationMV::MATOP_GET_DIAGONAL,
            // There are more
            _ => panic!("The given op: `{:?}` can not be turned into a `MatOperationMV`", op)
        }
    }
}

impl<'a, 'tl> Clone for Mat<'a, 'tl> {
    /// Same as [`x.duplicate(MatDuplicateOption::MAT_COPY_VALUES)`](Mat::duplicate()).
    fn clone(&self) -> Self {
        self.duplicate(MatDuplicateOption::MAT_COPY_VALUES).unwrap()
    }
}

impl<'a, 'tl, 'bv> BorrowMat<'a, 'tl, 'bv> {
    #[allow(dead_code)]
    pub(crate) fn new(owned_mat: ManuallyDrop<Mat<'a, 'tl>>, drop_func: Option<Box<dyn FnOnce(&mut BorrowMat<'a, 'tl, 'bv>) + 'bv>>) -> Self {
        BorrowMat { owned_mat, drop_func, _phantom: PhantomData }
    }
}

impl<'a, 'tl, 'bv> BorrowMatMut<'a, 'tl, 'bv> {
    #[allow(dead_code)]
    pub(crate) fn new(owned_mat: ManuallyDrop<Mat<'a, 'tl>>, drop_func: Option<Box<dyn FnOnce(&mut BorrowMatMut<'a, 'tl, 'bv>) + 'bv>>) -> Self {
        BorrowMatMut { owned_mat, drop_func, _phantom: PhantomData }
    }
}

impl<'a, 'tl> Deref for BorrowMat<'a, 'tl, '_> {
    type Target = Mat<'a, 'tl>;

    fn deref(&self) -> &Mat<'a, 'tl> {
        self.owned_mat.deref()
    }
}

impl<'a, 'tl> Deref for BorrowMatMut<'a, 'tl, '_> {
    type Target = Mat<'a, 'tl>;

    fn deref(&self) -> &Mat<'a, 'tl> {
        self.owned_mat.deref()
    }
}

impl<'a, 'tl> DerefMut for BorrowMatMut<'a, 'tl, '_> {
    fn deref_mut(&mut self) -> &mut Mat<'a, 'tl> {
        self.owned_mat.deref_mut()
    }
}

impl<'a> NullSpace<'a> {
    /// Creates a data structure used to project vectors out of null spaces.
    ///
    /// # Parameters
    ///
    /// * `world` - the MPI communicator associated with the object
    /// * `has_const` - if the null space contains the constant vector
    /// * `vecs` -  the vectors that span the null space (excluding
    /// the constant vector); these vectors must be orthonormal.
    ///
    /// Note, the "constant vector" is the vector with all entries being the same.
    pub fn create<T: Into<Vec<Vector<'a>>>>(world: &'a UserCommunicator, has_const: bool, vecs: T) -> Result<Self> {
        let vecs: Vec<Vector<'a>> = vecs.into();
        let vecs_p: Vec<_> = vecs.iter().map(|v| v.vec_p).collect();
        let n = vecs.len() as PetscInt;

        let mut ns_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::MatNullSpaceCreate(
            world.as_raw(), has_const.into(),
            n, vecs_p.as_ptr(), ns_p.as_mut_ptr()) };
        Petsc::check_error(world, ierr)?;

        Ok(NullSpace { world: world, ns_p: unsafe { ns_p.assume_init() } })
    }
}

// Macro impls
impl<'a> Mat<'a, '_> {    
    wrap_simple_petsc_member_funcs! {
        MatSetFromOptions, pub set_from_options, takes mut, #[doc = "Configures the Mat from the options database."];
        MatSetUp, pub set_up, takes mut, #[doc = "Sets up the internal matrix data structures for later use"];
        MatAssemblyBegin, pub assembly_begin, input MatAssemblyType, assembly_type, takes mut, #[doc = "Begins assembling the matrix. This routine should be called after completing all calls to MatSetValues()."];
        MatAssemblyEnd, pub assembly_end, input MatAssemblyType, assembly_type, takes mut, #[doc = "Completes assembling the matrix. This routine should be called after MatAssemblyBegin()."];
        MatGetLocalSize, pub get_local_size, output PetscInt, res1, output PetscInt, res2, #[doc = "Returns the number of local rows and local columns of a matrix.\n\nThat is the local size of the left and right vectors as returned by `MatCreateVecs()`"];
        MatGetSize, pub get_global_size, output PetscInt, res1, output PetscInt, res2, #[doc = "Returns the number of global rows and columns of a matrix."];
        MatMult, pub mult, input &Vector, x.as_raw, input &mut Vector, y.as_raw, #[doc = "Computes the matrix-vector product, y = Ax"];
        MatMultTranspose, pub mult_transpose, input &Vector, x.as_raw, input &mut Vector, y.as_raw, #[doc = "Computes matrix transpose times a vector y = A^T * x."];
        MatMultAdd, pub mult_add, input &Vector, v1.as_raw, input &Vector, v2.as_raw, input &mut Vector, v3.as_raw, #[doc = "Computes v3 = v2 + A * v1. "];
        MatMultTransposeAdd, pub mult_transpose_add, input &Vector, v1.as_raw, input &Vector, v2.as_raw, input &mut Vector, v3.as_raw, #[doc = "Computes v3 = v2 + A^T * v1."];
        MatNorm, pub norm, input NormType, norm_type, output PetscReal, tmp1, #[doc = "Calculates various norms of a matrix."];
        MatSetOption, pub set_option, input MatOption, option, input bool, flg, takes mut, #[doc = "Sets a parameter option for a matrix.\n\n\
            Some options may be specific to certain storage formats. Some options determine how values will be inserted (or added). Sorted, row-oriented input will generally assemble the fastest. The default is row-oriented."];
        MatGetDiagonal, pub get_diagonal, input &mut Vector, v.as_raw, #[doc = "Gets the diagonal of a matrix. "];
    }

    // TODO: there is more to each of these allocations that i should add support for
    wrap_prealloc_petsc_member_funcs! {
        MatSeqAIJSetPreallocation, seq_aij_set_preallocation, nz nz, nzz, #[doc = "For good matrix assembly \
            performance the user should preallocate the matrix storage by setting the parameter nz (or the array nnz). \
            By setting these parameters accurately, performance during matrix assembly can be increased by more than a \
            factor of 50.\n\n\
            Petsc C Docs: <https://petsc.org/release/docs/manualpages/Mat/MatSeqAIJSetPreallocation.html#MatSeqAIJSetPreallocation>\n\n\
            # Parameters.\n\n\
            * `nz` - number of nonzeros per row (same for all rows)\n\
            * `nnz` - slice containing the number of nonzeros in the various rows (possibly different for each row) or `None`"];
        MatSeqSELLSetPreallocation, seq_sell_set_preallocation, nz nz, nnz, #[doc = "For good matrix assembly \
            performance the user should preallocate the matrix storage by setting the parameter nz (or the array nnz). \
            By setting these parameters accurately, performance during matrix assembly can be increased significantly.\n\n\
            Petsc C Docs: <https://petsc.org/release/docs/manualpages/Mat/MatSeqSELLSetPreallocation.html#MatSeqSELLSetPreallocation>\n\n\
            # Parameters.\n\n\
            * `nz` - number of nonzeros per row (same for all rows)\n\
            * `nnz` - slice containing the number of nonzeros in the various rows (possibly different for each row) or `None`"];
        MatMPIAIJSetPreallocation, mpi_aij_set_preallocation, nz d_nz, d_nnz, nz o_nz, o_nnz, #[doc = "Preallocates memory for a \
        sparse parallel matrix in AIJ format (the default parallel PETSc format). For good matrix assembly performance the \
        user should preallocate the matrix storage by setting the parameters d_nz (or d_nnz) and o_nz (or o_nnz). By setting \
        these parameters accurately, performance can be increased by more than a factor of 50.\n\n\
        Petsc C Docs: <https://petsc.org/release/docs/manualpages/Mat/MatMPIAIJSetPreallocation.html#MatMPIAIJSetPreallocation>\n\n\
        # Parameters.\n\n\
        * `d_nz` - number of nonzeros per row in DIAGONAL portion of local submatrix (same value is used for all local rows)\n\
        * `d_nnz` - array containing the number of nonzeros in the various rows of the DIAGONAL portion of the local submatrix \
        (possibly different for each row) or `None`, if `d_nz` is used to specify the nonzero structure. The size of this array \
        is equal to the number of local rows, i.e `m`. For matrices that will be factored, you must leave room for (and set) the \
        diagonal entry even if it is zero.\n\
        * `o_nz` - number of nonzeros per row in the OFF-DIAGONAL portion of local submatrix (same value is used for all local rows).\n\
        * `o_nnz` - array containing the number of nonzeros in the various rows of the OFF-DIAGONAL portion of the local submatrix \
        (possibly different for each row) or `None`, if `o_nz` is used to specify the nonzero structure. The size of this array is \
        equal to the number of local rows, i.e 'm'."];
        MatMPISELLSetPreallocation, mpi_sell_set_preallocation, nz d_nz, d_nnz, nz o_nz, o_nnz, #[doc = "Preallocates memory for a \
        sparse parallel matrix in sell format. For good matrix assembly performance the user should preallocate the matrix storage \
        by setting the parameters `d_nz` (or `d_nnz`) and `o_nz` (or `o_nnz`).\n\n\
        Petsc C Docs: <https://petsc.org/release/docs/manualpages/Mat/MatMPISELLSetPreallocation.html#MatMPISELLSetPreallocation>\n\n\
        # Parameters.\n\n\
        Read docs for [`Mat::mpi_aij_set_preallocation()`](Mat::mpi_aij_set_preallocation())"];
        MatSeqSBAIJSetPreallocation, seq_sb_aij_set_preallocation, block bs, nz nz, nnz, #[doc = "Creates a sparse symmetric...\n\n\
        Petsc C Docs: <https://petsc.org/release/docs/manualpages/Mat/MatSeqSBAIJSetPreallocation.html#MatSeqSBAIJSetPreallocation>\n\n\
        # Parameters.\n\n\
        * `bs` - size of block, the blocks are ALWAYS square. One can use `MatSetBlockSizes()` to set a different row and column blocksize \
        but the row blocksize always defines the size of the blocks. The column blocksize sets the blocksize of the vectors obtained with `MatCreateVecs()`\n\
        * Read docs for [`Mat::seq_aij_set_preallocation()`](Mat::seq_aij_set_preallocation())"];
        MatMPISBAIJSetPreallocation, mpi_sb_aij_set_preallocation, block bs, nz d_nz, d_nnz, nz o_nz, o_nnz, #[doc = "For good matrix...\n\n\
        Petsc C Docs: <https://petsc.org/release/docs/manualpages/Mat/MatMPISBAIJSetPreallocation.html#MatMPISBAIJSetPreallocation>\n\n\
        # Parameters.\n\n\
        * `bs` - size of block, the blocks are ALWAYS square. One can use `MatSetBlockSizes()` to set a different row and column blocksize \
        but the row blocksize always defines the size of the blocks. The column blocksize sets the blocksize of the vectors obtained with `MatCreateVecs()`\n\
        * Read docs for [`Mat::mpi_aij_set_preallocation()`](Mat::mpi_aij_set_preallocation())"];
    }
}

impl_petsc_object_traits! { Mat, mat_p, petsc_raw::_p_Mat, '_ }

impl_petsc_view_func!{ Mat, MatView, '_ }

impl<'a> NullSpace<'a> {
    wrap_simple_petsc_member_funcs! {
        MatNullSpaceRemove, pub remove_from, input &mut Vector, vec.as_raw, #[doc = "Removes all the components of a null space from a vector."];
        MatNullSpaceTest, pub test, input &Mat, vec .as_raw, output bool, is_null .into from petsc_raw::PetscBool, #[doc = "Tests if the claimed null space is really a null space of a matrix."];
    }
}

impl_petsc_object_traits! { NullSpace, ns_p, petsc_raw::_p_MatNullSpace }

impl_petsc_view_func!{ NullSpace, MatNullSpaceView }
