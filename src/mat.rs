use crate::prelude::*;

// https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/index.html

// TODO: should we add a builder type so that you have to call some functions
// I feel like this could also be important for create, set up, assembly, and then finally using it.
// Because these stages need to be separate.
pub struct Mat<'a> {
    petsc: &'a crate::Petsc,
    pub(crate) mat_p: *mut petsc_raw::_p_Mat, // I could use Mat which is the same thing, but i think using a pointer is more clear
}

impl<'a> Drop for Mat<'a> {
    fn drop(&mut self) {
        // TODO: if the mat has more that one reference, than the object isn't really destroyed
        unsafe {
            let ierr = petsc_raw::MatDestroy(&mut self.mat_p as *mut *mut petsc_raw::_p_Mat);
            let _ = self.petsc.check_error(ierr); // TODO: should i unwrap or what idk?
        }
    }
}

pub use petsc_raw::MatAssemblyType;
pub use petsc_raw::MatOption;

impl_petsc_object_funcs!{ Mat, mat_p }

impl<'a> Mat<'a> {
    pub fn create(petsc: &'a crate::Petsc) -> Result<Self> {
        let mut mat_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::MatCreate(petsc.world.as_raw(), mat_p.as_mut_ptr()) };
        petsc.check_error(ierr)?;

        Ok(Mat { petsc, mat_p: unsafe { mat_p.assume_init() } })
    }

    /// Sets the local and global sizes, and checks to determine compatibility
    ///
    /// For rows and columns, local and global cannot be both None. If one processor calls this with a global of None then all processors must, otherwise the program will hang.
    /// If None is not used for the local sizes, then the user must ensure that they are chosen to be compatible with the vectors.
    pub fn set_sizes(&mut self, local_rows: Option<i32>, local_cols: Option<i32>, global_rows: Option<i32>, global_cols: Option<i32>) -> Result<()> {
        let ierr = unsafe { petsc_raw::MatSetSizes(
            self.mat_p, local_rows.unwrap_or(petsc_raw::PETSC_DECIDE_INTEGER), 
            local_cols.unwrap_or(petsc_raw::PETSC_DECIDE_INTEGER), 
            global_rows.unwrap_or(petsc_raw::PETSC_DECIDE_INTEGER), 
            global_cols.unwrap_or(petsc_raw::PETSC_DECIDE_INTEGER)) };
        self.petsc.check_error(ierr)
    }

    wrap_simple_petsc_member_funcs! {
        MatSetFromOptions, set_from_options, mat_p, #[doc = "Configures the Mat from the options database."];
        MatSetUp, set_up, mat_p, #[doc = "Sets up the internal matrix data structures for later use"];
    }
    
    // TODO: maybe these two functions should be combined with a lambda to run in between
    wrap_simple_petsc_member_funcs! {
        MatAssemblyBegin, assembly_begin, mat_p, assembly_type, MatAssemblyType, #[doc = "Begins assembling the matrix. This routine should be called after completing all calls to MatSetValues()."];
        MatAssemblyEnd, assembly_end, mat_p, assembly_type, MatAssemblyType, #[doc = "Completes assembling the matrix. This routine should be called after MatAssemblyBegin()."];
    }

    /// Inserts or adds a block of values into a matrix. These values may be cached, so MatAssemblyBegin()
    /// and MatAssemblyEnd() MUST be called after all calls to MatSetValues() have been completed.
    /// Read: <https://petsc.org/release/docs/manualpages/Mat/MatSetValues.html>
    pub fn set_values(&mut self, m: i32, idxm: &[i32], n: i32, idxn: &[i32], v: &[f64], addv: InsertMode) -> Result<()> {
        // TODO: I feel like most of the inputs are redundant and only will cause errors
        let ierr = unsafe { petsc_raw::MatSetValues(self.mat_p, m, idxm.as_ptr(), n, idxn.as_ptr(), v.as_ptr(), addv) };
        self.petsc.check_error(ierr)
    }

    /// Computes the matrix-vector product, y = Ax
    pub fn mult(&self, x: &Vector, y: &mut Vector) -> Result<()> {
        let ierr = unsafe { petsc_raw::MatMult(self.mat_p, x.vec_p, y.vec_p) };
        self.petsc.check_error(ierr)
    }

    /// Returns the range of matrix rows owned by this processor, assuming that the matrix is laid
    /// out with the first n1 rows on the first processor, the next n2 rows on the second, etc.
    /// For certain parallel layouts this range may not be well defined.
    pub fn get_ownership_range(&self) -> Result<std::ops::Range<i32>> {
        let mut low = MaybeUninit::<i32>::uninit();
        let mut high = MaybeUninit::<i32>::uninit();
        let ierr = unsafe { petsc_raw::MatGetOwnershipRange(self.mat_p, low.as_mut_ptr(), high.as_mut_ptr()) };
        self.petsc.check_error(ierr)?;

        Ok(unsafe { low.assume_init()..high.assume_init() })
    }

    /// Returns the range of matrix rows owned by EACH processor, assuming that the matrix is laid
    /// out with the first n1 rows on the first processor, the next n2 rows on the second, etc.
    /// For certain parallel layouts this range may not be well defined.
    pub fn get_ownership_ranges(&self) -> Result<Vec<std::ops::Range<i32>>> {
        let mut array = MaybeUninit::<*const i32>::uninit();
        let ierr = unsafe { petsc_raw::MatGetOwnershipRanges(self.mat_p, array.as_mut_ptr()) };
        self.petsc.check_error(ierr)?;

        // SAFETY: Petsc says it is an array of length size+1
        let slice_from_array = unsafe { 
            std::slice::from_raw_parts(array.assume_init(), self.petsc.world.size() as usize + 1) };
        let array_iter = slice_from_array.iter();
        let mut slice_iter_p1 = slice_from_array.iter();
        let _ = slice_iter_p1.next();
        Ok(array_iter.zip(slice_iter_p1).map(|(s,e)| *s..*e).collect())
    }

    /// Returns the number of local rows and local columns of a matrix, 
    /// that is the local size of the left and right vectors as returned by `MatCreateVecs()`
    pub fn get_local_size(&self) -> Result<(i32, i32)>
    {
        // Could this be a macro?
        let mut res1 = MaybeUninit::<i32>::uninit();
        let mut res2 = MaybeUninit::<i32>::uninit();
        let ierr = unsafe { petsc_raw::MatGetLocalSize(self.mat_p, res1.as_mut_ptr(), res2.as_mut_ptr()) };
        self.petsc.check_error(ierr)?;

        Ok(unsafe { (res1.assume_init(), res2.assume_init()) })
    }

    /// Sets a parameter option for a matrix. Some options may be specific to certain storage formats. 
    /// Some options determine how values will be inserted (or added). Sorted, row-oriented input will
    /// generally assemble the fastest. The default is row-oriented.
    pub fn set_option(&mut self, option: MatOption, flg: bool) -> Result<()>
    {
        let ierr = unsafe { petsc_raw::MatSetOption(self.mat_p, 
            option, if flg {petsc_raw::PetscBool::PETSC_TRUE} else {petsc_raw::PetscBool::PETSC_FALSE}) };
        self.petsc.check_error(ierr)
    }

    // TODO: there is more to each of these allocations that i should add support for
    wrap_prealloc_petsc_member_funcs! {
        MatSeqAIJSetPreallocation, seq_aij_set_preallocation, mat_p, nz, nzz, #[doc = "For good matrix assembly \
            performance the user should preallocate the matrix storage by setting the parameter nz (or the array nnz). \
            By setting these parameters accurately, performance during matrix assembly can be increased by more than a \
            factor of 50.\n\n\
            Petsc C Docs: <https://petsc.org/release/docs/manualpages/Mat/MatSeqAIJSetPreallocation.html#MatSeqAIJSetPreallocation>\n\n\
            Parameters.\n\n\
            * `nz` - number of nonzeros per row (same for all rows)\n\
            * `nnz` - slice containing the number of nonzeros in the various rows (possibly different for each row) or `None`"];
        MatSeqSELLSetPreallocation, seq_sell_set_preallocation, mat_p, nz, nnz, #[doc = "For good matrix assembly \
            performance the user should preallocate the matrix storage by setting the parameter nz (or the array nnz). \
            By setting these parameters accurately, performance during matrix assembly can be increased significantly.\n\n\
            Petsc C Docs: <https://petsc.org/release/docs/manualpages/Mat/MatSeqSELLSetPreallocation.html#MatSeqSELLSetPreallocation>\n\n\
            Parameters.\n\n\
            * `nz` - number of nonzeros per row (same for all rows)\n\
            * `nnz` - slice containing the number of nonzeros in the various rows (possibly different for each row) or `None`"];
    }

    wrap_prealloc_petsc_member_funcs! {
        MatMPIAIJSetPreallocation, mpi_aij_set_preallocation, mat_p, d_nz, d_nnz, o_nz, o_nnz, #[doc = "Preallocates memory for a \
        sparse parallel matrix in AIJ format (the default parallel PETSc format). For good matrix assembly performance the \
        user should preallocate the matrix storage by setting the parameters d_nz (or d_nnz) and o_nz (or o_nnz). By setting \
        these parameters accurately, performance can be increased by more than a factor of 50.\n\n\
        Petsc C Docs: <https://petsc.org/release/docs/manualpages/Mat/MatMPIAIJSetPreallocation.html#MatMPIAIJSetPreallocation>\n\n\
        Parameters.\n\n\
        * `d_nz` - number of nonzeros per row in DIAGONAL portion of local submatrix (same value is used for all local rows)\n\
        * `d_nnz` - array containing the number of nonzeros in the various rows of the DIAGONAL portion of the local submatrix \
        (possibly different for each row) or `None`, if `d_nz` is used to specify the nonzero structure. The size of this array \
        is equal to the number of local rows, i.e `m`. For matrices that will be factored, you must leave room for (and set) the \
        diagonal entry even if it is zero.\n\
        * `o_nz` - number of nonzeros per row in the OFF-DIAGONAL portion of local submatrix (same value is used for all local rows).\n\
        * `o_nnz` - array containing the number of nonzeros in the various rows of the OFF-DIAGONAL portion of the local submatrix \
        (possibly different for each row) or `None`, if `o_nz` is used to specify the nonzero structure. The size of this array is \
        equal to the number of local rows, i.e 'm'."];
        MatMPISELLSetPreallocation, mpi_sell_set_preallocation, mat_p, d_nz, d_nnz, o_nz, o_nnz, #[doc = "Preallocates memory for a \
        sparse parallel matrix in sell format. For good matrix assembly performance the user should preallocate the matrix storage \
        by setting the parameters `d_nz` (or `d_nnz`) and `o_nz` (or `o_nnz`).\n\n\
        Petsc C Docs: <https://petsc.org/release/docs/manualpages/Mat/MatMPISELLSetPreallocation.html#MatMPISELLSetPreallocation>\n\n\
        Parameters.\n\n\
        Read docs for [`Mat::mpi_aij_set_preallocation()`](Mat::mpi_aij_set_preallocation())"];
    }

    wrap_prealloc_petsc_member_funcs! {
        MatSeqSBAIJSetPreallocation, seq_sb_aij_set_preallocation, mat_p, bs, nz, nnz, #[doc = "Creates a sparse symmetric...\n\n\
        Petsc C Docs: <https://petsc.org/release/docs/manualpages/Mat/MatSeqSBAIJSetPreallocation.html#MatSeqSBAIJSetPreallocation>\n\n\
        Parameters.\n\n\
        * `bs` - size of block, the blocks are ALWAYS square. One can use `MatSetBlockSizes()` to set a different row and column blocksize \
        but the row blocksize always defines the size of the blocks. The column blocksize sets the blocksize of the vectors obtained with `MatCreateVecs()`\n\
        * Read docs for [`Mat::seq_aij_set_preallocation()`](Mat::seq_aij_set_preallocation())"];
    }

    wrap_prealloc_petsc_member_funcs! {
        MatMPISBAIJSetPreallocation, mpi_sb_aij_set_preallocation, mat_p, bs, d_nz, d_nnz, o_nz, o_nnz, #[doc = "For good matrix...\n\n\
        Petsc C Docs: <https://petsc.org/release/docs/manualpages/Mat/MatMPISBAIJSetPreallocation.html#MatMPISBAIJSetPreallocation>\n\n\
        Parameters.\n\n\
        * `bs` - size of block, the blocks are ALWAYS square. One can use `MatSetBlockSizes()` to set a different row and column blocksize \
        but the row blocksize always defines the size of the blocks. The column blocksize sets the blocksize of the vectors obtained with `MatCreateVecs()`\n\
        * Read docs for [`Mat::mpi_aij_set_preallocation()`](Mat::mpi_aij_set_preallocation())"];
    }
}
