//! PETSc vectors (Vec objects) are used to store the field variables in PDE-based (or other) simulations.
//!
//! PETSc C API docs: <https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/index.html>

use crate::prelude::*;

// TODO: should we add a builder type so that you have to call set_type or set_from_options in order to use the vector
/// Abstract PETSc vector object
pub struct Vector<'a> {
    petsc: &'a crate::Petsc,

    pub(crate) vec_p: *mut petsc_raw::_p_Vec, // I could use Vec which is the same thing, but i think using a pointer is more clear
}

impl<'a> Drop for Vector<'a> {
    fn drop(&mut self) {
        unsafe {
            let ierr = petsc_raw::VecDestroy(&mut self.vec_p as *mut *mut petsc_raw::_p_Vec);
            let _ = self.petsc.check_error(ierr); // TODO should i unwrap or what idk?
        }
    }
}

impl_petsc_object_funcs!{ Vector, vec_p }

pub use petsc_raw::NormType;

impl<'a> Vector<'a> {
    /// Creates an empty vector object. The type can then be set with [`Vector::set_type`](#), or [`Vector::set_from_options`].
    /// Same as [`Petsc::vec_create`].
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// let petsc = Petsc::init_no_args().unwrap();
    ///
    /// Vector::create(&petsc).unwrap();
    /// ```
    pub fn create(petsc: &'a crate::Petsc) -> Result<Self> {
        let mut vec_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::VecCreate(petsc.world.as_raw(), vec_p.as_mut_ptr()) };
        petsc.check_error(ierr)?;

        Ok(Vector { petsc, vec_p: unsafe { vec_p.assume_init() } })
    }

    /// Creates a new vector of the same type as an existing vector.
    /// [`duplicate`](Vector::duplicate) DOES NOT COPY the vector entries, but rather 
    /// allocates storage for the new vector. Use [`Vector::copy_values`](#) to copy a vector.
    pub fn duplicate(&self) -> Result<Self> {
        let mut vec2_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::VecDuplicate(self.vec_p, vec2_p.as_mut_ptr()) };
        self.petsc.check_error(ierr)?;

        Ok(Vector { petsc: &self.petsc, vec_p: unsafe { vec2_p.assume_init() } })
    }

    wrap_simple_petsc_member_funcs! {
        VecSetFromOptions, set_from_options, vec_p, #[doc = "Configures the vector from the options database."];
        VecSetUp, set_up, vec_p, #[doc = "Sets up the internal vector data structures for the later use."];
        VecAssemblyBegin, assembly_begin, vec_p, #[doc = "Begins assembling the vector. This routine should be called after completing all calls to VecSetValues()."];
        VecAssemblyEnd, assembly_end, vec_p, #[doc = "Completes assembling the vector. This routine should be called after VecAssemblyBegin()."];
    }

    wrap_simple_petsc_member_funcs! {
        VecSet, set_all, vec_p, alpha, f64, #[doc = "Sets all components of a vector to a single scalar value.\n\nYou CANNOT call this after you have called [`Vector::set_values()`]."];
    }

    wrap_simple_petsc_member_funcs! {
        VecGetLocalSize, get_local_size, vec_p, i32, #[doc = "Returns the number of elements of the vector stored in local memory."];
        VecGetSize, get_global_size, vec_p, i32, #[doc = "Returns the global number of elements of the vector."];
    }

    ///  Assembling the vector by calling [`Vector::assembly_begin()`] then [`Vector::assembly_end()`]
    pub fn assemble(&mut self) -> Result<()>
    {
        self.assembly_begin()?;
        // TODO: what would even go here?
        self.assembly_end()
    }

    /// Sets the local and global sizes, and checks to determine compatibility
    /// The inputs can be `None` to have PETSc decide the size.
    /// `local_size` and `global_size` cannot be both `None`. If one processor calls this with
    /// `global_size` of `None` then all processors must, otherwise the program will hang.
    pub fn set_sizes(&mut self, local_size: Option<i32>, global_size: Option<i32>) -> Result<()> {
        let ierr = unsafe { petsc_raw::VecSetSizes(
            self.vec_p, local_size.unwrap_or(petsc_raw::PETSC_DECIDE_INTEGER), 
            global_size.unwrap_or(petsc_raw::PETSC_DECIDE_INTEGER)) };
        self.petsc.check_error(ierr)
    }

    /// Computes self += alpha * other
    pub fn axpy(&mut self, alpha: f64, other: &Vector) -> Result<()>
    {
        let ierr = unsafe { petsc_raw::VecAXPY(self.vec_p, alpha, other.vec_p) };
        self.petsc.check_error(ierr)
    }

    /// Computes the vector norm.
    pub fn norm(&self, norm_type: NormType) -> Result<f64>
    {
        let mut res = MaybeUninit::<f64>::uninit();
        let ierr = unsafe { petsc_raw::VecNorm(self.vec_p, norm_type, res.as_mut_ptr()) };
        self.petsc.check_error(ierr)?;

        Ok(unsafe { res.assume_init() })
    }

    /// Inserts or adds values into certain locations of a vector.
    /// `ix` and `v` must be the same length or `set_values` will panic
    ///
    /// Parameters.
    /// 
    /// * `ix` - The relative convergence tolerance, relative decrease in the (possibly preconditioned) residual norm
    /// * `v` - The absolute convergence tolerance absolute size of the (possibly preconditioned) residual norm
    /// * `iora` - Either [`INSERT_VALUES`](InsertMode::INSERT_VALUES) or [`ADD_VALUES`](InsertMode::ADD_VALUES), 
    /// where [`ADD_VALUES`](InsertMode::ADD_VALUES) adds values to any existing entries, and 
    /// [`INSERT_VALUES`](InsertMode::INSERT_VALUES) replaces existing entries with new values.
    ///
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # let petsc = Petsc::init_no_args().unwrap();
    /// let mut v = petsc.vec_create().unwrap();
    /// v.set_sizes(None, Some(10)).unwrap(); // create vector of size 10
    /// v.set_from_options().unwrap();
    ///
    /// v.set_values(&[0, 3, 7, 9], &[1.1, 2.2, 3.3, 4.4], InsertMode::INSERT_VALUES).unwrap();
    /// assert_eq!(&v.get_values(0..10).unwrap()[..], &[1.1,0.0,0.0,2.2,0.0,0.0,0.0,3.3,0.0,4.4]);
    ///
    /// v.set_values(&vec![0, 2, 8, 9], &vec![1.0, 2.0, 3.0, 4.0], InsertMode::ADD_VALUES).unwrap();
    /// assert_eq!(&v.get_values(0..10).unwrap()[..], &[2.1,0.0,2.0,2.2,0.0,0.0,0.0,3.3,3.0,8.4]);
    /// ```
    pub fn set_values(&mut self, ix: &[i32], v: &[f64], iora: InsertMode) -> Result<()>
    {
        // TODO: should I do these asserts?
        assert!(iora == InsertMode::INSERT_VALUES || iora == InsertMode::ADD_VALUES);
        assert_eq!(ix.len(), v.len());

        let ni = ix.len() as i32;
        let ierr = unsafe { petsc_raw::VecSetValues(self.vec_p, ni, ix.as_ptr(), v.as_ptr(), iora) };
        self.petsc.check_error(ierr)
    }

    /// Gets values from certain locations of a vector. Currently can only get values on the same processor
    ///
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # let petsc = Petsc::init_no_args().unwrap();
    /// let mut v = petsc.vec_create().unwrap();
    /// v.set_sizes(None, Some(10)).unwrap(); // create vector of size 10
    /// v.set_from_options().unwrap();
    ///
    /// let ix = [0, 2, 7, 9];
    /// v.set_values(&ix, &[1.1, 2.2, 3.3, 4.4], InsertMode::INSERT_VALUES).unwrap();
    ///
    /// assert_eq!(&v.get_values(ix).unwrap()[..], &[1.1, 2.2, 3.3, 4.4]);
    /// assert_eq!(&v.get_values(vec![2, 0, 9, 7]).unwrap()[..], &[2.2, 1.1, 4.4, 3.3]);
    /// assert_eq!(&v.get_values(0..10).unwrap()[..], &[1.1,0.0,2.2,0.0,0.0,0.0,0.0,3.3,0.0,4.4]);
    /// assert_eq!(&v.get_values((0..5).map(|v| v*2)).unwrap()[..], &[1.1,2.2,0.0,0.0,0.0]);
    /// ```
    pub fn get_values<T>(&self, ix: T) -> Result<Vec<f64>>
    where
        T: IntoIterator<Item = i32>,
        <T as IntoIterator>::IntoIter: ExactSizeIterator
    {
        // TODO: ix can't be a &[i32; N] because it impl IntoIterator with Item=&i32 (same with &Vec<i32>)
        // This might not be an issue because [i32; N] implements copy.

        // TODO: make this return a slice (like how libceed has the VectorView type)
        // For now im going to return a vector because this is a temporary testing function
        // Really we need to use `VecView` which can get data from the whole vector

        // TODO: is this good, it feels like it would be better to just accept &[i32] and then we dont 
        // need to do the collect. Although, it is nice to accept ranges or iters as input.

        let ix_iter = ix.into_iter();
        let ni = ix_iter.len();
        let ix_array = ix_iter.collect::<Vec<_>>();
        let mut out_vec = vec![f64::default();ni];

        let ierr = unsafe { petsc_raw::VecGetValues(self.vec_p, ni as i32, ix_array.as_ptr(), out_vec[..].as_mut_ptr()) };
        self.petsc.check_error(ierr)?;

        Ok(out_vec)
    }

    /// Returns the range of indices owned by this processor, assuming that the vectors are laid
    /// out with the first n1 elements on the first processor, next n2 elements on the second, etc.
    /// For certain parallel layouts this range may not be well defined.
    pub fn get_ownership_range(&self) -> Result<std::ops::Range<i32>> {
        let mut low = MaybeUninit::<i32>::uninit();
        let mut high = MaybeUninit::<i32>::uninit();
        let ierr = unsafe { petsc_raw::VecGetOwnershipRange(self.vec_p, low.as_mut_ptr(), high.as_mut_ptr()) };
        self.petsc.check_error(ierr)?;

        Ok(unsafe { low.assume_init()..high.assume_init() })
    }

    /// Returns the range of indices owned by EACH processor, assuming that the vectors are laid
    /// out with the first n1 elements on the first processor, next n2 elements on the second, etc.
    /// For certain parallel layouts this range may not be well defined.
    pub fn get_ownership_ranges(&self) -> Result<Vec<std::ops::Range<i32>>> {
        let mut array = MaybeUninit::<*const i32>::uninit();
        let ierr = unsafe { petsc_raw::VecGetOwnershipRanges(self.vec_p, array.as_mut_ptr()) };
        self.petsc.check_error(ierr)?;

        // SAFETY: Petsc says it is an array of length size+1
        let slice_from_array = unsafe { 
            std::slice::from_raw_parts(array.assume_init(), self.petsc.world.size() as usize + 1) };
        let array_iter = slice_from_array.iter();
        let mut slice_iter_p1 = slice_from_array.iter();
        let _ = slice_iter_p1.next();
        Ok(array_iter.zip(slice_iter_p1).map(|(s,e)| *s..*e).collect())
    }

    // TODO: add `from_array`/`from_slice` and maybe also `set_slice`
}

impl_petsc_view_func!{ Vector, vec_p, VecView }
