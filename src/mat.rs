//! PETSc matrices (Mat objects) are used to store Jacobians and other
//! sparse matrices in PDE-based (or other) simulations.
//!
//! PETSc C API docs: <https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/index.html>

use crate::prelude::*;

// TODO: should we add a builder type so that you have to call some functions
// I feel like this could also be important for create, set up, assembly, and then finally using it.
// Because these stages need to be separate.

/// Abstract PETSc matrix object used to manage all linear operators in PETSc, even those
/// without an explicit sparse representation (such as matrix-free operators).
pub struct Mat<'a> {
    pub(crate) world: &'a dyn Communicator,
    pub(crate) mat_p: *mut petsc_raw::_p_Mat, // I could use Mat which is the same thing, but i think using a pointer is more clear
}

impl<'a> Drop for Mat<'a> {
    fn drop(&mut self) {
        unsafe {
            let ierr = petsc_raw::MatDestroy(&mut self.mat_p as *mut *mut petsc_raw::_p_Mat);
            let _ = Petsc::check_error(self.world, ierr); // TODO: should i unwrap or what idk?
        }
    }
}

/// Abstract PETSc object that removes a null space from a vector, i.e. orthogonalizes the vector to a subspace.
pub struct NullSpace<'a> {
    pub(crate) world: &'a dyn Communicator,
    pub(crate) ns_p: *mut petsc_raw::_p_MatNullSpace,
}

impl Drop for NullSpace<'_> {
    fn drop(&mut self) {
        unsafe {
            let ierr = petsc_raw::MatNullSpaceDestroy(&mut self.ns_p as *mut _);
            let _ = Petsc::check_error(self.world, ierr); // TODO: should i unwrap or what idk?
        }
    }
}

pub use petsc_raw::MatAssemblyType;
pub use petsc_raw::MatOption;
pub use petsc_raw::MatDuplicateOption;
pub use petsc_raw::MatStencil;
use petsc_raw::MatReuse;

impl<'a> Mat<'a> {
    /// Same at [`Petsc::mat_create()`].
    pub fn create(world: &'a dyn Communicator,) -> Result<Self> {
        let mut mat_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::MatCreate(world.as_raw(), mat_p.as_mut_ptr()) };
        Petsc::check_error(world, ierr)?;

        Ok(Mat { world, mat_p: unsafe { mat_p.assume_init() } })
    }

    /// Duplicates a matrix including the non-zero structure.
    ///
    /// Note, [`Mat::clone()`] is the same as `x.duplicate(MatDuplicateOption::MAT_COPY_VALUES)`.
    ///
    /// See the manual page for [`MatDuplicateOption`](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatDuplicateOption.html#MatDuplicateOption) for an explanation of these options.
    pub fn duplicate(&self, op: MatDuplicateOption) -> Result<Self> {
        let mut mat2_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::MatDuplicate(self.mat_p, op, mat2_p.as_mut_ptr()) };
        Petsc::check_error(self.world, ierr)?;

        Ok(Mat { world: self.world, mat_p: unsafe { mat2_p.assume_init() } })
    }

    /// Sets the local and global sizes, and checks to determine compatibility
    ///
    /// For rows and columns, local and global cannot be both None. If one processor calls this with a global of None then all processors must, otherwise the program will hang.
    /// If None is not used for the local sizes, then the user must ensure that they are chosen to be compatible with the vectors.
    pub fn set_sizes(&mut self, local_rows: Option<i32>, local_cols: Option<i32>, global_rows: Option<i32>, global_cols: Option<i32>) -> Result<()> {
        let ierr = unsafe { petsc_raw::MatSetSizes(
            self.mat_p, local_rows.unwrap_or(petsc_raw::PETSC_DECIDE), 
            local_cols.unwrap_or(petsc_raw::PETSC_DECIDE), 
            global_rows.unwrap_or(petsc_raw::PETSC_DECIDE), 
            global_cols.unwrap_or(petsc_raw::PETSC_DECIDE)) };
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
    /// You might find [`Mat::assemble_with()`] or [`Mat::assemble_with_batched()`] more useful and
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
    /// # fn main() -> petsc_rs::Result<()> {
    /// # let petsc = Petsc::init_no_args().unwrap();
    /// if petsc.world().size() != 1 {
    ///     // note, cargo wont run tests with mpi so this will never be reached,
    ///     // but this example will only work in a uniprocessor comm world
    ///     Petsc::set_error(petsc.world(), PetscErrorKind::PETSC_ERROR_WRONG_MPI_SIZE, "This is a uniprocessor example only!")?;
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
    ///     let v = [i as f64 * 3.0, i as f64 * 3.0+1.0, i as f64 * 3.0+2.0];
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
    ///     let v = [i as f64 , i as f64 + 3.0, i as f64 + 6.0];
    ///     mat2.set_values(&[0,1,2], &[i], &v, InsertMode::INSERT_VALUES)?;
    /// }
    /// // You MUST assemble before you can use 
    /// mat2.assembly_begin(MatAssemblyType::MAT_FINAL_ASSEMBLY)?;
    /// mat2.assembly_end(MatAssemblyType::MAT_FINAL_ASSEMBLY)?;
    /// # // for debugging
    /// # mat2.view_with(Some(&viewer))?;
    /// 
    /// assert_eq!(&mat.get_values(0..n, 0..n).unwrap()[..], &mat2.get_values(0..n, 0..n).unwrap()[..]);
    /// assert_eq!(&mat.get_values(0..n, 0..n).unwrap()[..], &[ 0.0,  1.0,  2.0,
    ///                                                         3.0,  4.0,  5.0,
    ///                                                         6.0,  7.0,  8.0,]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_values(&mut self, idxm: &[i32], idxn: &[i32], v: &[f64], addv: InsertMode) -> Result<()> {
        let m = idxm.len();
        let n = idxn.len();
        assert_eq!(v.len(), m*n);
        let ierr = unsafe { petsc_raw::MatSetValues(self.mat_p, m as i32, idxm.as_ptr(), n as i32,
            idxn.as_ptr(), v.as_ptr(), addv) };
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
    /// C API docs: <https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatSetValuesStencil.html#MatSetValuesStencil>
    ///
    /// # Notes for [`MatStencil`] type
    ///
    /// The `i`,`j`, and `k` represent the logical coordinates over the entire grid (for 2 and
    /// 1 dimensional problems the `k` and `j` entries are ignored). The `c` represents the the degrees
    /// of freedom at each grid point (the dof argument to DMDASetDOF()). If dof is 1 then this entry
    /// is ignored.
    pub fn set_values_stencil(&mut self, idxm: &[MatStencil], idxn: &[MatStencil], v: &[f64], addv: InsertMode) -> Result<()> {
        let m = idxm.len();
        let n = idxn.len();
        assert_eq!(v.len(), m*n);
        let ierr = unsafe { petsc_raw::MatSetValuesStencil(self.mat_p, m as i32, idxm.as_ptr(), n as i32,
            idxn.as_ptr(), v.as_ptr(), addv) };
        Petsc::check_error(self.world, ierr)
    }

    /// Returns the range of matrix rows owned by this processor, assuming that the matrix is laid
    /// out with the first n1 rows on the first processor, the next n2 rows on the second, etc.
    /// For certain parallel layouts this range may not be well defined.
    pub fn get_ownership_range(&self) -> Result<std::ops::Range<i32>> {
        let mut low = MaybeUninit::<i32>::uninit();
        let mut high = MaybeUninit::<i32>::uninit();
        let ierr = unsafe { petsc_raw::MatGetOwnershipRange(self.mat_p, low.as_mut_ptr(), high.as_mut_ptr()) };
        Petsc::check_error(self.world, ierr)?;

        Ok(unsafe { low.assume_init()..high.assume_init() })
    }

    /// Returns the range of matrix rows owned by EACH processor, assuming that the matrix is laid
    /// out with the first n1 rows on the first processor, the next n2 rows on the second, etc.
    /// For certain parallel layouts this range may not be well defined.
    pub fn get_ownership_ranges(&self) -> Result<Vec<std::ops::Range<i32>>> {
        let mut array = MaybeUninit::<*const i32>::uninit();
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
    /// from [`Mat::set_values`], returning it.
    ///
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # fn main() -> petsc_rs::Result<()> {
    /// # let petsc = Petsc::init_no_args().unwrap();
    /// if petsc.world().size() != 1 {
    ///     // note, cargo wont run tests with mpi so this will never be reached,
    ///     // but this example will only work in a uniprocessor comm world
    ///     Petsc::set_error(petsc.world(), PetscErrorKind::PETSC_ERROR_WRONG_MPI_SIZE, "This is a uniprocessor example only!")?;
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
    ///         .map(|(i,j)| if i == j { (i, j, 2.0) }
    ///                      else { (i, j, -1.0) }),
    ///     InsertMode::INSERT_VALUES, MatAssemblyType::MAT_FINAL_ASSEMBLY)?;
    /// # // for debugging
    /// # let viewer = Viewer::create_ascii_stdout(petsc.world())?;
    /// # mat.view_with(Some(&viewer))?;
    /// 
    /// assert_eq!(&mat.get_values(0..n, 0..n).unwrap()[..], &[ 2.0, -1.0,  0.0,  0.0,  0.0,
    ///                                                        -1.0,  2.0, -1.0,  0.0,  0.0,
    ///                                                         0.0, -1.0,  2.0, -1.0,  0.0,
    ///                                                         0.0,  0.0, -1.0,  2.0, -1.0,
    ///                                                         0.0,  0.0,  0.0, -1.0,  2.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn assemble_with<I>(&mut self, iter_builder: I, addv: InsertMode, assembly_type: MatAssemblyType) -> Result<()>
    where
        I: IntoIterator<Item = (i32, i32, f64)>
    {
        // We don't actually care about the num_inserts value, we just need something that
        // implements `Sum` so we can use the sum method and `()` does not.
        let _num_inserts = iter_builder.into_iter().map(|(idxm, idxn, v)| {
            self.set_values(std::slice::from_ref(&idxm), std::slice::from_ref(&idxn),
                std::slice::from_ref(&v), addv).map(|_| 1)
        }).sum::<Result<i32>>()?;
        // Note, `sum()` will short-circuit the iterator if an error is encountered.

        self.assembly_begin(assembly_type)?;
        self.assembly_end(assembly_type)
    }

    /// Allows you to give an iter that will be use to make a series of calls to [`Mat::set_values`].
    /// Then is followed by both [`Mat::assembly_begin()`] and [`Mat::assembly_end()`].
    ///
    /// Unlike [`assemble_with()`](Mat::assemble_with()), this will set values in batches, i.e., the
    /// input is given with array like values. They can be [`array`]s, [`slice`]s, [`Vec`]s, or anything that
    /// implements [`AsRef<[...]>`](AsRef).
    ///
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # fn main() -> petsc_rs::Result<()> {
    /// # let petsc = Petsc::init_no_args().unwrap();
    /// if petsc.world().size() != 1 {
    ///     // note, cargo wont run tests with mpi so this will never be reached,
    ///     // but this example will only work in a uniprocessor comm world
    ///     Petsc::set_error(petsc.world(), PetscErrorKind::PETSC_ERROR_WRONG_MPI_SIZE, "This is a uniprocessor example only!")?;
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
    /// let v = [-1.0, 2.0, -1.0];
    /// mat.assemble_with_batched((0..n)
    ///         .map(|i| if i == 0 { ( vec![i], vec![i, i+1], &v[1..] ) }
    ///                  else if i == n { ( vec![i], vec![i-1, i], &v[..2] ) }
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
    ///                                                   0.0,  0.0,  0.0, -1.0,  2.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn assemble_with_batched<I, A1, A2, A3>(&mut self, iter_builder: I, addv: InsertMode, assembly_type: MatAssemblyType) -> Result<()>
    where
        I: IntoIterator<Item = (A1, A2, A3)>,
        A1: AsRef<[i32]>,
        A2: AsRef<[i32]>,
        A3: AsRef<[f64]>,
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
        I: IntoIterator<Item = (MatStencil, MatStencil, f64)>
    {
        // We don't actually care about the num_inserts value, we just need something that
        // implements `Sum` so we can use the sum method and `()` does not.
        let _num_inserts = iter_builder.into_iter().map(|(idxm, idxn, v)| {
            self.set_values_stencil(std::slice::from_ref(&idxm), std::slice::from_ref(&idxn),
                std::slice::from_ref(&v), addv).map(|_| 1)
        }).sum::<Result<i32>>()?;
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
        A3: AsRef<[f64]>,
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

    /// Gets values from certain locations of a Matrix. Currently can only get values on the same processor.
    ///
    /// # Example
    /// 
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # fn main() -> petsc_rs::Result<()> {
    /// # let petsc = Petsc::init_no_args().unwrap();
    /// if petsc.world().size() != 1 {
    ///     // note, cargo wont run tests with mpi so this will never be reached,
    ///     // but this example will only work in a uniprocessor comm world
    ///     Petsc::set_error(petsc.world(), PetscErrorKind::PETSC_ERROR_WRONG_MPI_SIZE, "This is a uniprocessor example only!")?;
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
    ///         .map(|(v, (i,j))| (i,j,v as f64)),
    ///     InsertMode::INSERT_VALUES, MatAssemblyType::MAT_FINAL_ASSEMBLY)?;
    /// # // for debugging
    /// # let viewer = Viewer::create_ascii_stdout(petsc.world())?;
    /// # mat.view_with(Some(&viewer))?;
    /// 
    /// assert_eq!(&mat.get_values(0..n, 0..n).unwrap()[..], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    /// assert_eq!(&mat.get_values(0..n, [0]).unwrap()[..], &[0.0, 3.0, 6.0]);
    /// assert_eq!(&mat.get_values([1], 0..2).unwrap()[..], &[3.0, 4.0]);
    /// assert_eq!(&mat.get_values([1,2], [0,2]).unwrap()[..], &[3.0, 5.0, 6.0, 8.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_values<T1, T2>(&self, idxm: T1, idxn: T2) -> Result<Vec<f64>>
    where
        T1: IntoIterator<Item = i32>,
        <T1 as IntoIterator>::IntoIter: ExactSizeIterator,
        T2: IntoIterator<Item = i32>,
        <T2 as IntoIterator>::IntoIter: ExactSizeIterator,
    {
        let idxm_iter = idxm.into_iter();
        let mi = idxm_iter.len();
        let idxm_array = idxm_iter.collect::<Vec<_>>();

        let idxn_iter = idxn.into_iter();
        let ni = idxn_iter.len();
        let idxn_array = idxn_iter.collect::<Vec<_>>();
        
        let mut out_vec = vec![f64::default(); mi * ni];

        let ierr = unsafe { petsc_raw::MatGetValues(self.mat_p, mi as i32, idxm_array.as_ptr(),
            ni as i32, idxn_array.as_ptr(), out_vec[..].as_mut_ptr()) };
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

    /// Performs Matrix-Matrix Multiplication C=A*B.
    ///
    /// 
    pub fn mat_mult(&self, other: &Mat, fill: Option<f64>) -> Result<Mat<'a>>{
        let mut mat_out_p = MaybeUninit::uninit();
        // TODO: do we want other MatReuse options
        let ierr = unsafe { petsc_raw::MatMatMult(self.mat_p, other.mat_p, MatReuse::MAT_INITIAL_MATRIX,
            fill.unwrap_or(petsc_raw::PETSC_DEFAULT_REAL), mat_out_p.as_mut_ptr()) };
        Petsc::check_error(self.world, ierr)?;

        Ok(Mat { world: self.world, mat_p: unsafe { mat_out_p.assume_init() } })
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
    pub fn set_nullspace(&mut self, nullspace: Option<Rc<NullSpace<'a>>>) -> Result<()> {
        let ierr = unsafe { petsc_raw::MatSetNullSpace(self.mat_p,
            nullspace.as_ref().map_or(std::ptr::null_mut(), |ns| ns.ns_p)) };
        Petsc::check_error(self.world, ierr)
    }

    /// Attaches a left null space to a matrix.
    ///
    /// This null space is used by the linear solvers. Overwrites any previous null space that may
    /// have been attached. You can remove the null space by calling this routine with `None`.
    ///
    /// Krylov solvers can produce the minimal norm solution to the least squares problem by utilizing
    /// [`NullSpace::remove_from()`].
    // TODO: I don't like this api, we only force it to be a `Rc` because we dont want the caller editing
    // the nullspace after they set it (we don't actually use it to reference count).
    pub fn set_left_nullspace(&mut self, nullspace: Option<Rc<NullSpace<'a>>>) -> Result<()> {
        let ierr = unsafe { petsc_raw::MatSetTransposeNullSpace(self.mat_p,
            nullspace.as_ref().map_or(std::ptr::null_mut(), |ns| ns.ns_p)) };
        Petsc::check_error(self.world, ierr)
    }

    /// Attaches a null space to a matrix, which is often the null space (rigid body modes)
    /// of the operator without boundary conditions This null space will be used to provide
    /// near null space vectors to a multigrid preconditioner built from this matrix.
    ///
    /// This null space is used by the linear solvers. Overwrites any previous null space that may
    /// have been attached. You can remove the null space by calling this routine with `None`.
    // TODO: I don't like this api, we only force it to be a `Rc` because we dont want the caller editing
    // the nullspace after they set it (we don't actually use it to reference count).
    pub fn set_near_nullspace(&mut self, nullspace: Option<Rc<NullSpace<'a>>>) -> Result<()> {
        let ierr = unsafe { petsc_raw::MatSetNearNullSpace(self.mat_p,
            nullspace.as_ref().map_or(std::ptr::null_mut(), |ns| ns.ns_p)) };
        Petsc::check_error(self.world, ierr)
    }
}

impl Clone for Mat<'_> {
    /// Same as [`x.duplicate(MatDuplicateOption::MAT_COPY_VALUES)`](Mat::duplicate()).
    fn clone(&self) -> Self {
        self.duplicate(MatDuplicateOption::MAT_COPY_VALUES).unwrap()
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
    pub fn create<T: Into<Vec<Vector<'a>>>>(world: &'a dyn Communicator, has_const: bool, vecs: T) -> Result<Self> {
        let vecs: Vec<Vector<'a>> = vecs.into();
        let vecs_p: Vec<_> = vecs.iter().map(|v| v.vec_p).collect();
        let n = vecs.len() as i32;

        let mut ns_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::MatNullSpaceCreate(
            world.as_raw(), has_const.into(),
            n, vecs_p.as_ptr(), ns_p.as_mut_ptr()) };
        Petsc::check_error(world, ierr)?;

        Ok(NullSpace { world: world, ns_p: unsafe { ns_p.assume_init() } })
    }
}

// Macro impls
impl<'a> Mat<'a> {    
    wrap_simple_petsc_member_funcs! {
        MatSetFromOptions, set_from_options, mat_p, takes mut, #[doc = "Configures the Mat from the options database."];
        MatSetUp, set_up, mat_p, takes mut, #[doc = "Sets up the internal matrix data structures for later use"];
        MatAssemblyBegin, assembly_begin, mat_p, input MatAssemblyType, assembly_type, takes mut, #[doc = "Begins assembling the matrix. This routine should be called after completing all calls to MatSetValues()."];
        MatAssemblyEnd, assembly_end, mat_p, input MatAssemblyType, assembly_type, takes mut, #[doc = "Completes assembling the matrix. This routine should be called after MatAssemblyBegin()."];
        MatGetLocalSize, get_local_size, mat_p, output i32, res1, output i32, res2, #[doc = "Returns the number of local rows and local columns of a matrix.\n\nThat is the local size of the left and right vectors as returned by `MatCreateVecs()`"];
        MatGetSize, get_global_size, mat_p, output i32, res1, output i32, res2, #[doc = "Returns the number of global rows and columns of a matrix."];
        MatMult, mult, mat_p, input &Vector, x.vec_p, input &mut Vector, y.vec_p, #[doc = "Computes the matrix-vector product, y = Ax"];
        MatNorm, norm, mat_p, input NormType, norm_type, output f64, tmp1, #[doc = "Calculates various norms of a matrix."];
        MatSetOption, set_option, mat_p, input MatOption, option, input bool, flg, #[doc = "Sets a parameter option for a matrix.\n\n\
            Some options may be specific to certain storage formats. Some options determine how values will be inserted (or added). Sorted, row-oriented input will generally assemble the fastest. The default is row-oriented."];
    }

    // TODO: there is more to each of these allocations that i should add support for
    wrap_prealloc_petsc_member_funcs! {
        MatSeqAIJSetPreallocation, seq_aij_set_preallocation, mat_p, nz nz, nzz, #[doc = "For good matrix assembly \
            performance the user should preallocate the matrix storage by setting the parameter nz (or the array nnz). \
            By setting these parameters accurately, performance during matrix assembly can be increased by more than a \
            factor of 50.\n\n\
            Petsc C Docs: <https://petsc.org/release/docs/manualpages/Mat/MatSeqAIJSetPreallocation.html#MatSeqAIJSetPreallocation>\n\n\
            # Parameters.\n\n\
            * `nz` - number of nonzeros per row (same for all rows)\n\
            * `nnz` - slice containing the number of nonzeros in the various rows (possibly different for each row) or `None`"];
        MatSeqSELLSetPreallocation, seq_sell_set_preallocation, mat_p, nz nz, nnz, #[doc = "For good matrix assembly \
            performance the user should preallocate the matrix storage by setting the parameter nz (or the array nnz). \
            By setting these parameters accurately, performance during matrix assembly can be increased significantly.\n\n\
            Petsc C Docs: <https://petsc.org/release/docs/manualpages/Mat/MatSeqSELLSetPreallocation.html#MatSeqSELLSetPreallocation>\n\n\
            # Parameters.\n\n\
            * `nz` - number of nonzeros per row (same for all rows)\n\
            * `nnz` - slice containing the number of nonzeros in the various rows (possibly different for each row) or `None`"];
        MatMPIAIJSetPreallocation, mpi_aij_set_preallocation, mat_p, nz d_nz, d_nnz, nz o_nz, o_nnz, #[doc = "Preallocates memory for a \
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
        MatMPISELLSetPreallocation, mpi_sell_set_preallocation, mat_p, nz d_nz, d_nnz, nz o_nz, o_nnz, #[doc = "Preallocates memory for a \
        sparse parallel matrix in sell format. For good matrix assembly performance the user should preallocate the matrix storage \
        by setting the parameters `d_nz` (or `d_nnz`) and `o_nz` (or `o_nnz`).\n\n\
        Petsc C Docs: <https://petsc.org/release/docs/manualpages/Mat/MatMPISELLSetPreallocation.html#MatMPISELLSetPreallocation>\n\n\
        # Parameters.\n\n\
        Read docs for [`Mat::mpi_aij_set_preallocation()`](Mat::mpi_aij_set_preallocation())"];
        MatSeqSBAIJSetPreallocation, seq_sb_aij_set_preallocation, mat_p, block bs, nz nz, nnz, #[doc = "Creates a sparse symmetric...\n\n\
        Petsc C Docs: <https://petsc.org/release/docs/manualpages/Mat/MatSeqSBAIJSetPreallocation.html#MatSeqSBAIJSetPreallocation>\n\n\
        # Parameters.\n\n\
        * `bs` - size of block, the blocks are ALWAYS square. One can use `MatSetBlockSizes()` to set a different row and column blocksize \
        but the row blocksize always defines the size of the blocks. The column blocksize sets the blocksize of the vectors obtained with `MatCreateVecs()`\n\
        * Read docs for [`Mat::seq_aij_set_preallocation()`](Mat::seq_aij_set_preallocation())"];
        MatMPISBAIJSetPreallocation, mpi_sb_aij_set_preallocation, mat_p, block bs, nz d_nz, d_nnz, nz o_nz, o_nnz, #[doc = "For good matrix...\n\n\
        Petsc C Docs: <https://petsc.org/release/docs/manualpages/Mat/MatMPISBAIJSetPreallocation.html#MatMPISBAIJSetPreallocation>\n\n\
        # Parameters.\n\n\
        * `bs` - size of block, the blocks are ALWAYS square. One can use `MatSetBlockSizes()` to set a different row and column blocksize \
        but the row blocksize always defines the size of the blocks. The column blocksize sets the blocksize of the vectors obtained with `MatCreateVecs()`\n\
        * Read docs for [`Mat::mpi_aij_set_preallocation()`](Mat::mpi_aij_set_preallocation())"];
    }
}

impl_petsc_object_funcs!{ Mat, mat_p }

impl_petsc_view_func!{ Mat, mat_p, MatView }

impl<'a> NullSpace<'a> {
    wrap_simple_petsc_member_funcs! {
        MatNullSpaceRemove, remove_from, ns_p, input &mut Vector, vec.vec_p, #[doc = "Removes all the components of a null space from a vector."];
        MatNullSpaceTest, test, ns_p, input &Mat, vec .mat_p, output bool, is_null .into from petsc_raw::PetscBool, #[doc = "Tests if the claimed null space is really a null space of a matrix."];
    }
}

impl_petsc_object_funcs!{ NullSpace, ns_p }

impl_petsc_view_func!{ NullSpace, ns_p, MatNullSpaceView }
