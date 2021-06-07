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
    pub fn set_values(&mut self, m: i32, idxm: &Vec<i32>, n: i32, idxn: &Vec<i32>, v: &Vec<f64>, addv: InsertMode) -> Result<()> {
        // TODO: I feel like most of the inputs are redundant and only will cause errors
        let ierr = unsafe { petsc_raw::MatSetValues(self.mat_p, m, idxm.as_ptr(), n, idxn.as_ptr(), v.as_ptr(), addv) };
        self.petsc.check_error(ierr)
    }

    /// Computes the matrix-vector product, y = Ax
    pub fn mult(&self, x: &Vector, y: &mut Vector) -> Result<()> {
        let ierr = unsafe { petsc_raw::MatMult(self.mat_p, x.vec_p, y.vec_p) };
        self.petsc.check_error(ierr)
    }

}
